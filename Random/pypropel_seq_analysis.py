#%%
import pypropel as pp
import tmkit as tmk
import numpy as np
import pandas as pd
import requests
import time
# import sys
# import sqlite3



# %%
datapath = '/Users/jasonschmidberger/DataScience/data'

df_old = pp.io.read(
    df_fpn=f'{datapath}/pdbtm_alpha.txt',
    # df_fpn=pdb_fpn_dict['10.02.2023'],
    # df_fpn=pdb_fpn_dict['12.06.2024'],
    df_sep='\t',
    header=None,
)
df_old

# %%
df_short = df_old.head(200)
df_short
# %%
df = pd.DataFrame()
df['prot'] = df_short[0].astype(str).str[:4]
df['chain'] = df_short[0].astype(str).str[-1:]
df['ij'] = df.apply(lambda x: x['prot'] + '.' + x['chain'], axis=1)
print(df)
prot_series = pd.Series(df['prot'].unique())
print(prot_series)

# %%
tmk.seq.retrieve_pdb_from_pdbtm(
    prot_series=prot_series,
    kind='tr',
    sv_fp=f'{datapath}/cplx/',
)


def run_psiblast(fasta_str: str, top_n: int = 10, poll_interval: int = 6, max_attempts: int = 30) -> pd.DataFrame:
    """
    Submit a PSI-BLAST job to NCBI, wait for completion, and return the top N hits as a DataFrame.
    """
    if not fasta_str.strip().startswith('>'):
        raise ValueError('FASTA sequence must include a header line starting with ">".')

    put_params = {
        'CMD': 'Put',
        'PROGRAM': 'blastp',
        'SERVICE': 'PsiBlast',
        'DATABASE': 'nr',
        'HITLIST_SIZE': str(top_n),
        'FORMAT_TYPE': 'JSON2',
        'QUERY': fasta_str,
    }
    put_resp = requests.post('https://blast.ncbi.nlm.nih.gov/Blast.cgi', data=put_params, timeout=30)
    put_resp.raise_for_status()

    rid = None
    rtoe = poll_interval
    for line in put_resp.text.splitlines():
        if line.startswith('RID ='):
            rid = line.split('=')[1].strip()
        elif line.startswith('RTOE ='):
            try:
                rtoe = max(poll_interval, int(line.split('=')[1].strip()))
            except ValueError:
                pass
    if not rid:
        raise RuntimeError('Failed to obtain RID for PSI-BLAST submission.')

    time.sleep(rtoe)
    for attempt in range(max_attempts):
        get_params = {
            'CMD': 'Get',
            'FORMAT_TYPE': 'JSON2',
            'RID': rid,
            'SERVICE': 'PsiBlast',
        }
        get_resp = requests.get('https://blast.ncbi.nlm.nih.gov/Blast.cgi', params=get_params, timeout=30)
        get_resp.raise_for_status()
        status_text = get_resp.text

        if 'Status=READY' in status_text and 'ThereAreHits=yes' in status_text:
            json_resp = requests.get(
                'https://blast.ncbi.nlm.nih.gov/Blast.cgi',
                params={**get_params, 'FORMAT_OBJECT': 'Alignment', 'FORMAT_TYPE': 'JSON2'},
                timeout=30,
            )
            json_resp.raise_for_status()
            search_info = json_resp.json()
            hits = search_info.get('BlastOutput2', [{}])[0].get('report', {}).get('results', {}).get('search', {}).get('hits', [])
            rows = []
            for hit in hits[:top_n]:
                hsp = hit.get('hsps', [{}])[0]
                rows.append(
                    {
                        'hit_id': hit.get('description', [{}])[0].get('accession', ''),
                        'hit_def': hit.get('description', [{}])[0].get('title', ''),
                        'evalue': hsp.get('evalue', None),
                        'bit_score': hsp.get('bit_score', None),
                        'identity_pct': (hsp.get('identity', 0) / max(hsp.get('align_len', 1), 1)) * 100,
                        'align_len': hsp.get('align_len', None),
                    }
                )
            return pd.DataFrame(rows)

        if 'Status=READY' in status_text and 'ThereAreHits=no' in status_text:
            return pd.DataFrame(columns=['hit_id', 'hit_def', 'evalue', 'bit_score', 'identity_pct', 'align_len'])

        if 'Status=FAILED' in status_text:
            raise RuntimeError(f'PSI-BLAST job {rid} failed.')
        time.sleep(poll_interval)

    raise TimeoutError(f'PSI-BLAST job {rid} did not complete after {max_attempts} attempts.')


def fetch_pdb_fasta(pdb_code: str, chain: str | None = None) -> str:
    """
    Fetch the FASTA sequence for a PDB entry (optionally a specific chain).

    Examples
    --------
    fetch_pdb_fasta("1A0I")              -> whole entry
    fetch_pdb_fasta("1A0I", chain="A")   -> chain A only
    """
    pdb_code = pdb_code.upper()
    if chain:
        chain = chain.upper()
        url = f"https://www.rcsb.org/fasta/entry/{pdb_code}/download"
        params = {"structureId": f"{pdb_code}.{chain}"}
    else:
        url = f"https://www.rcsb.org/fasta/entry/{pdb_code}/download"
        params = None

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.text.strip()
# %%
fasta = fetch_pdb_fasta("3BJX", chain="A")
fasta
#%%
hits_df = run_psiblast(fasta, top_n=10)
print(hits_df)


# %%
