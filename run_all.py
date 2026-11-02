"""Executa o pipeline completo (CRISP-DM) na ordem correta.

    python run_all.py                        # tudo
    python run_all.py --skip-ednet-extract   # reaproveita data/processed/ednet_users.csv.gz (extração leva ~10 min)
    python run_all.py --notebooks            # depois do pipeline, executa os notebooks 01–09 (salvando as saídas)

Se o EdNet-KT3 (zip ou pasta) não for encontrado (ver src/config.py) mas data/processed/ednet_users.csv.gz existir,
a extração é pulada com um aviso e as análises usam o arquivo processado.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
import config as C  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

steps = ["ares_01_prepare.py", "ednet_01_extract.py", "ares_02_analise.py", "ednet_02_analise.py", "comparacao.py",
         "ml_ednet_abandono.py", "ml_ares_desempenho.py",   # 2º bimestre: aprendizado de máquina
         "auditoria.py"]

processado = C.PROCESSED / "ednet_users.csv.gz"
if "--skip-ednet-extract" in sys.argv:
    steps.remove("ednet_01_extract.py")
elif not C.EDNET_BRUTO_DISPONIVEL:
    if processado.exists():
        print("AVISO: EdNet bruto (EdNet-KT3 + EdNet-Contents) não encontrado. Pastas procuradas:\n         "
              + "\n         ".join(str(b) for b in C._BASES) + "\n"
              f"       Pulando a extração e usando {processado.relative_to(ROOT)} (já processado).\n"
              "       Para reprocessar, coloque EdNet-KT3 (zip ou pasta) e EdNet-Contents em data/raw/ednet "
              "ou defina a variável EDNET_DIR.", flush=True)
        steps.remove("ednet_01_extract.py")
    else:
        sys.exit(f"ERRO: EdNet bruto não encontrado (pastas procuradas: {[str(b) for b in C._BASES]}) "
                 f"e não há {processado.relative_to(ROOT)}.\n"
                 "Baixe os dados (seção 11 de docs/DOCUMENTACAO.md) para data/raw/ednet ou defina EDNET_DIR.")

# UTF-8 nos subprocessos: evita acentos corrompidos no console do Windows
env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
for s in steps:
    print(f"\n=== {s} ===", flush=True)
    r = subprocess.run([sys.executable, str(SRC / s)], env=env)
    if r.returncode != 0:
        sys.exit(f"\nERRO: {s} terminou com código {r.returncode}. O traceback acima mostra a causa "
                 "(a última linha é a mensagem de erro; os sinais ~~~^^^ só apontam o trecho da linha).")
print("\nPipeline concluído.")

if "--notebooks" in sys.argv:
    # executa os notebooks na ordem, salvando as saídas no próprio arquivo (requer: pip install nbconvert ipykernel)
    for nb in sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb")):
        print(f"\n=== notebook {nb.name} ===", flush=True)
        r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                            "--ExecutePreprocessor.timeout=3600", nb.name], cwd=nb.parent, env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(r.stderr[-3000:])
            sys.exit(f"\nERRO no notebook {nb.name}. Abra-o no VS Code/Jupyter para ver a célula que falhou.")
        print("ok", flush=True)
    print("\nNotebooks executados.")
