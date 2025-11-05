import os, subprocess, json, pathlib, sys

def run(cmd, **kw):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)

def structure_guided_msa(query_fasta, workdir="sg_msa", use_templates=True):
    wd = pathlib.Path(workdir); wd.mkdir(parents=True, exist_ok=True)

    # 1) Get homologs via ColabFold MMseqs2 server → A3M
    a3m = wd / "query.a3m"
    run(["colabfold_search", "--use-templates", "0", query_fasta, str(wd)])
    # ColabFold writes {basename}.a3m; normalize name:
    found = list(wd.glob("*.a3m"))
    assert found, "No A3M produced by colabfold_search"
    found[0].rename(a3m)

    # 2) Get a query structure (fastest: AlphaFold DB via UniProt ID),
    # or generate with colabfold_batch for the query alone.
    # Here we assume you predicted one already at wd/query_model.pdb
    query_pdb = wd / "query_model.pdb"
    if not query_pdb.exists():
        print("WARNING: query_model.pdb not found – skipping Foldseek structural search on query.")
        print("You can still run Expresso with template list derived from sequence→PDB mapping.")
    
    # 3) Use Foldseek to find PDB templates (needs query structure)
    templates_tsv = wd / "pdb_hits.tsv"
    if query_pdb.exists():
        # DB choices: 'pdb' (Foldseek's prebuilt) or your local path
        # Ensure foldseek databases are set up; replace 'pdb' with your DB path if needed.
        run([
            "foldseek", "easy-search",
            str(query_pdb),
            "pdb",  # requires foldseek databases installed/configured
            str(templates_tsv),
            str(wd / "tmp"),
            "--format-output", "target,alntmscore,lddt,taln,tseq,tacc"
        ])
    else:
        open(templates_tsv, "w").close()

    # 4) Build template mapping for Expresso (seqID  PDB:XXXX_A)
    # For simplicity, we convert Foldseek hits into a template list for the query only.
    # In a more thorough pipeline, you would also map top homolog sequences to their PDBs.
    template_list = wd / "templates.list"
    with open(template_list, "w") as out:
        if templates_tsv.stat().st_size == 0:
            print("No structural templates from Foldseek; Expresso can still fetch by itself.")
        else:
            with open(templates_tsv) as f:
                for i, line in enumerate(f):
                    parts = line.strip().split('\t')
                    if len(parts) < 6: continue
                    target = parts[0]  # typically like pdb|XXXX|A or similar
                    # Extract PDB ID / chain heuristically:
                    pdbid, chain = None, None
                    if '|' in target:
                        toks = target.split('|')
                        if len(toks) >= 3:
                            pdbid = toks[1]; chain = toks[2]
                    # Fallback parse (Foldseek formats can vary)
                    if pdbid and chain:
                        out.write(f"query PDB:{pdbid}_{chain}\n")

    # 5) Write a FASTA containing query + A3M sequences (A3M → FASTA ungapped for uppercase)
    fasta_for_expresso = wd / "expresso_input.fa"
    def a3m_to_fasta(a3m_path, out_fa):
        seqs = []
        with open(a3m_path) as f:
            name, seq = None, []
            for line in f:
                if line.startswith('>'):
                    if name:
                        seqs.append((name, ''.join(seq)))
                        seq = []
                    name = line[1:].strip().split()[0]
                else:
                    seq.append(''.join([c for c in line.strip() if not c.islower()]))  # drop inserts
            if name:
                seqs.append((name, ''.join(seq)))
        with open(out_fa, "w") as o:
            for n, s in seqs:
                o.write(f">{n}\n{s}\n")
    a3m_to_fasta(a3m, fasta_for_expresso)

    # 6) Run T-Coffee in structure mode (Expresso). It can fetch PDB automatically.
    # If you have a local PDB mirror, add: -pdb_db /path/to/pdb
    # If you want to enforce your templates: -pdb_templatefile templates.list
    out_aln = wd / "msa_expresso.aln"
    cmd = [
        "t_coffee",
        "-mode", "expresso",
        "-seq", str(fasta_for_expresso),
        "-output", "fasta_aln,clustalw_aln",
        "-outfile", str(out_aln)
    ]
    if template_list.exists() and template_list.stat().st_size > 0:
        cmd += ["-pdb_templatefile", str(template_list)]
    run(cmd)

    print(f"\nDone. Structure-guided MSA at: {out_aln}")
    return out_aln

# Example:
# structure_guided_msa("query.fasta", workdir="sg_msa")
