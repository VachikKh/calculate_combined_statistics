import pytest

def test_combined_mean_std():
    x1 = np.random.normal(loc=15, size = 5005)
    x2 = np.random.rand(1200)
    x3 = np.random.normal(loc = 115, scale = 2, size = 1150)

    x_i = np.c_[[x1, x2, x3]]

    x = np.concatenate([x1, x2, x3])

    m1, s1, n1 = x1.mean(), x1.std(), x1.shape[0]
    m2, s2, n2 = x2.mean(), x2.std(), x2.shape[0]
    m3, s3, n3 = x3.mean(), x3.std(), x3.shape[0]
    m, s, n  = x.mean(), x.std(), x.shape[0]

    m_i = np.array([m1, m2, m3])
    s_i = np.array([s1, s2, s3])
    n_i = np.array([n1, n2, n3])
    cm, cs = combined_mean_std(m_i, s_i, n_i)
    assert m==pytest.approx(cm)
    assert s == pytest.approx(cs)
    return True
