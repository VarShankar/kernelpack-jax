import numpy as np
from kernelpack.geometry.cfpu import _winding_sign_reliable


def test_winding_guard_accepts_clear_signs_and_rejects_ambiguous_estimates():
    coarse=np.array([.147,.853,.02,.98,.49,-1.,.1,np.nan,np.inf])
    fine=np.array([.025,.975,.01,.99,.99,.02,.3,.01,.01])
    expected=np.array([True,True,True,True,False,False,False,False,False])
    np.testing.assert_array_equal(_winding_sign_reliable(coarse,fine,.1),expected)
