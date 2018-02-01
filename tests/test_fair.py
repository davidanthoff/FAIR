import pytest

import fair
import os
import numpy as np
import warnings
from fair.RCPs import rcp3pd, rcp45, rcp6, rcp85
from fair.constants import molwt
from fair.tools import magicc


def test_import():
    version = fair.__version__
    assert version[:3] == '1.1'


def test_no_arguments():
    with pytest.raises(ValueError):
        fair.forward.fair_scm()


def test_zero_emissions():
    nt = 250
    emissions = np.zeros(nt)
    C,F,T = fair.forward.fair_scm(emissions=emissions, other_rf=0.)
    assert np.allclose(C,np.ones(nt)*278.)
    assert np.allclose(F,np.zeros(nt))
    assert np.allclose(T,np.zeros(nt))


def test_ten_GtC_pulse():
    emissions = np.zeros(250)
    emissions[125:] = 10.0
    other_rf = np.zeros(emissions.size)
    for x in range(0,emissions.size):
        other_rf[x] = 0.5*np.sin(2*np.pi*(x)/14.0)

    C,F,T = fair.forward.fair_scm(emissions=emissions, other_rf=other_rf)

    datadir = os.path.join(os.path.dirname(__file__), 'ten_GtC_pulse/')
    C_expected = np.load(datadir + 'C.npy')
    F_expected = np.load(datadir + 'F.npy')
    T_expected = np.load(datadir + 'T.npy')

    assert np.allclose(C, C_expected)
    assert np.allclose(F, F_expected)
    assert np.allclose(T, T_expected)


def test_multigas_fullemissions_error():
    with pytest.raises(ValueError):
        fair.forward.fair_scm(emissions=rcp3pd.Emissions.emissions)


# There must be a good way to avoid duplication here
def test_rcp3pd():
    C,F,T = fair.forward.fair_scm(emissions=rcp3pd.Emissions.emissions,
        useMultigas=True)

    datadir = os.path.join(os.path.dirname(__file__), 'rcp3pd/')
    C_expected = np.load(datadir + 'C.npy')
    F_expected = np.load(datadir + 'F.npy')
    T_expected = np.load(datadir + 'T.npy')

    assert np.allclose(C, C_expected)
    assert np.allclose(F, F_expected)
    assert np.allclose(T, T_expected)


def test_rcp45():
    C,F,T = fair.forward.fair_scm(emissions=rcp45.Emissions.emissions,
        useMultigas=True)

    datadir = os.path.join(os.path.dirname(__file__), 'rcp45/')
    C_expected = np.load(datadir + 'C.npy')
    F_expected = np.load(datadir + 'F.npy')
    T_expected = np.load(datadir + 'T.npy')

    assert np.allclose(C, C_expected)
    assert np.allclose(F, F_expected)
    assert np.allclose(T, T_expected)


def test_rcp6():
    C,F,T = fair.forward.fair_scm(emissions=rcp6.Emissions.emissions,
        useMultigas=True)

    datadir = os.path.join(os.path.dirname(__file__), 'rcp6/')
    C_expected = np.load(datadir + 'C.npy')
    F_expected = np.load(datadir + 'F.npy')
    T_expected = np.load(datadir + 'T.npy')

    assert np.allclose(C, C_expected)
    assert np.allclose(F, F_expected)
    assert np.allclose(T, T_expected)


def test_rcp85():
    C,F,T = fair.forward.fair_scm(emissions=rcp85.Emissions.emissions,
        useMultigas=True)

    datadir = os.path.join(os.path.dirname(__file__), 'rcp85/')
    C_expected = np.load(datadir + 'C.npy')
    F_expected = np.load(datadir + 'F.npy')
    T_expected = np.load(datadir + 'T.npy')

    assert np.allclose(C, C_expected)
    assert np.allclose(F, F_expected)
    assert np.allclose(T, T_expected)


def test_division():
    # Ensure parameters given as integers are treated as floats when dividing
    # (Python2 compatibility).
    _, _, T = fair.forward.fair_scm(
        emissions=fair.RCPs.rcp6.Emissions.emissions,
        useMultigas=True,
        d=np.array([239.0, 4.0]),
        tcr_dbl=70.0
    )
    _, _, T_int_params = fair.forward.fair_scm(
        emissions=fair.RCPs.rcp6.Emissions.emissions,
        useMultigas=True,
        d=np.array([239, 4]),
        tcr_dbl=70
    )
    assert (T == T_int_params).all()


def test_scenfile():
    datadir = os.path.join(os.path.dirname(__file__), 'rcp45/')
    # Purpose of this test is to determine whether the SCEN file for RCP4.5
    # which does not include CFCs, years before 2000, or emissions from every 
    # year from 2000 to 2500, equals the emissions file from RCP4.5 
    # after reconstruction.
    # The .SCEN and .XLS files at http://www.pik-potsdam.de/~mmalte/rcps
    # sometimes differ in the 4th decimal place. Thus we allow a tolerance of 
    # 0.0002 in addition to machine error in this instance.

    E1 = magicc.scen_open(datadir + 'RCP45.SCEN')
    assert np.allclose(E1, rcp45.Emissions.emissions, rtol=1e-8, atol=2e-4)

    E2 = magicc.scen_open(datadir + 'RCP45.SCEN', include_cfcs=False)
    assert np.allclose(E2, rcp45.Emissions.emissions[:,:24], rtol=1e-8,
        atol=2e-4)

    E3 = magicc.scen_open(datadir + 'RCP45.SCEN', startyear=1950)
    assert np.allclose(E3, rcp45.Emissions.emissions[185:,:], rtol=1e-8,
        atol=2e-4)


# Between v1.1 and v1.2 we switched from specifying aviation NOx as a 
# fraction of the total NOx emissions to an absolute (Mt NO2/yr) value.
# This checks that the same input gives the same output, and that other
# basic checks (aviNOx <= total NOx and no aviNOx = no contrail forcing),
# depreciation warning for aviNOx frac) are satisfied.
def test_aviNOx_frac_deprec(recwarn):
    warnings.simplefilter('always', DeprecationWarning)
    C, F, T = fair.forward.fair_scm(
        emissions=fair.RCPs.rcp85.Emissions.emissions,
        useMultigas=True,
        aviNOx_frac=fair.RCPs.rcp85.aviNOx_frac)
    assert len(recwarn)==2
    assert recwarn.pop(DeprecationWarning)
    assert recwarn.pop(UserWarning)


def test_aviNOx_frac_abs():
    datadir = os.path.join(os.path.dirname(__file__), 'aviNOx/')
    C_expected = np.load(datadir + 'C_CO2.npy')
    F_expected = np.load(datadir + 'F_con.npy')
    T_expected = np.load(datadir + 'T.npy')

    C, F, T = fair.forward.fair_scm(
        emissions=fair.RCPs.rcp85.Emissions.emissions,
        useMultigas=True,
        aviNOx=fair.RCPs.rcp85.aviNOx)

    # because of rounding errors results are slightly different from the old
    # treatment.
    assert np.allclose(C[:,0], C_expected) 
    assert np.allclose(F[:,7], F_expected, atol=5e-4)
    assert np.allclose(T, T_expected, atol=2e-4)


def test_aviNOx_zero():
    _, F, _ = fair.forward.fair_scm(rcp85.Emissions.emissions,
                     useMultigas=True,
                     aviNOx=0.)
    assert (F[:,7] == np.zeros_like(F[:,7])).all()


def test_aviNOx_gt_total():
    with pytest.raises(ValueError):
        C, F, T = fair.forward.fair_scm(
            emissions=fair.RCPs.rcp85.Emissions.emissions,
            useMultigas=True,
            aviNOx=fair.RCPs.rcp85.Emissions.nox*1.01*molwt.NO2/molwt.N)
