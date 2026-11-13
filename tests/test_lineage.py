"""Linhagem: md5 do dado (ponteiro DVC ou CSV em disco) e commit do código."""
import hashlib

from churn.lineage import UNKNOWN, data_md5, git_commit


def test_data_md5_le_o_ponteiro_dvc(tmp_path):
    ponteiro = tmp_path / "churn.csv.dvc"
    ponteiro.write_text("outs:\n- md5: abc123\n  size: 10\n  path: churn.csv\n")
    assert data_md5(ponteiro) == "abc123"


def test_data_md5_sem_ponteiro_usa_o_csv(tmp_path):
    """Sem ponteiro DVC (é o caso deste repositório), hasheia o CSV em disco.

    Sem esse fallback a tag `data_md5` de todo run sairia "unknown" e a
    linhagem do experimento ficaria inútil.
    """
    csv = tmp_path / "churn.csv"
    csv.write_bytes(b"customerID,Churn\n1,No\n")
    esperado = hashlib.md5(csv.read_bytes()).hexdigest()
    assert data_md5(tmp_path / "nao_existe.dvc", csv) == esperado


def test_data_md5_sem_ponteiro_nem_csv_nao_quebra(tmp_path):
    assert data_md5(tmp_path / "nao_existe.dvc", tmp_path / "nao_existe.csv") == UNKNOWN


def test_data_md5_ignora_ponteiro_corrompido(tmp_path):
    """Ponteiro em YAML inválido não pode derrubar o treino."""
    ponteiro = tmp_path / "quebrado.dvc"
    ponteiro.write_text("isto: nao: e: um: dvc valido")
    csv = tmp_path / "churn.csv"
    csv.write_bytes(b"a,b\n1,2\n")
    assert data_md5(ponteiro, csv) == hashlib.md5(csv.read_bytes()).hexdigest()


def test_git_commit_devolve_string_nao_vazia():
    assert isinstance(git_commit(), str) and git_commit()
