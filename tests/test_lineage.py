from churn.lineage import UNKNOWN, data_md5, git_commit


def test_data_md5_le_o_ponteiro_dvc(tmp_path):
    ponteiro = tmp_path / "churn.csv.dvc"
    ponteiro.write_text("outs:\n- md5: abc123\n  size: 10\n  path: churn.csv\n")
    assert data_md5(ponteiro) == "abc123"


def test_data_md5_sem_ponteiro_nao_quebra(tmp_path):
    assert data_md5(tmp_path / "nao_existe.dvc") == UNKNOWN


def test_git_commit_devolve_string_nao_vazia():
    assert isinstance(git_commit(), str) and git_commit()
