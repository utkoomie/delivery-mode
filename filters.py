#------------------------------------------------------------------
# Support utilities for machine-learning analysis using CDC data
# 
# Copyright 2018-2022 Karl W. Schulz
# 
# Dell Medical School, University of Texas
#------------------------------------------------------------------
import utils
import pandas

#--
# apply data filtering based on keys defined in the choices list. Filtering is applied in list order.
def apply_filtering(data,choices,header=True):

    if header:
        utils.pdCount(data,'[Starting]')

    for filter in choices:
        if filter == 'unrevised_certs':
            start = data.shape[0]
            data.drop(data[(data['revision'] != 'A')].index,inplace=True)
            data.drop(data[(data.me_trial == -999)].index,inplace=True)
            utils.pdCount(data,comment='Drop unrevised certificates',start=start)
        elif filter ==  'rdmeth_rec_empty':
            start = data.shape[0]
            data = data.drop(data[(data.rdmeth_rec == ' ')].index); 
            utils.pdCount(data,comment='Drop unknown rdmeth_rec',start=start)
        elif filter == 'rdmeth_rec_to_int':
            data = data.astype({"rdmeth_rec": int})
        elif filter == 'umhisp_to_int':
            data = data.astype({"umhisp": int})
        elif filter == 'rdmeth_rec_4known_methods':
            start = data.shape[0]
            data = data.drop(data[(data.rdmeth_rec > 4)].index)
            utils.pdCount(data,comment='Drop rdmeth_rec > 4',start=start)
        elif filter == 'rdmeth_rec_unknown':
            start = data.shape[0]
            data = data.drop(data[(data.rdmeth_rec < 1)].index)
            utils.pdCount(data,comment='Drop rdmeth_rec < 1',start=start)
        elif filter == 'rdmeth_rec_planned_c_sections':
            start = data.shape[0]
            data = data.drop(data[(data.rdmeth_rec == 3) & (data.me_trial == 'N')].index)
            data = data.drop(data[(data.rdmeth_rec == 4) & (data.me_trial == 'N')].index)
            utils.pdCount(data,comment='Drop C-sects w/ trial of labor == no',start=start)
        elif filter == 'rdmeth_rec_unknown_me_trial':
            start = data.shape[0]
            data = data.drop(data[(data.rdmeth_rec == 3) & ~( (data.me_trial == 'Y') | (data.me_trial == 'N') )].index)
            data = data.drop(data[(data.rdmeth_rec == 4) & ~( (data.me_trial == 'Y') | (data.me_trial == 'N') )].index)
            utils.pdCount(data,comment='Drop C-sects w/ unknown trial of labor',start=start)
        elif filter == 'me_trial_empty':
            start = data.shape[0]
            data = data.drop(data[(data.me_trial == ' ')].index)
            utils.pdCount(data,comment='Drop empty me_trial',start=start)
        elif filter == 'me_trial_unknown':
            start = data.shape[0]
            data = data.drop(data[(data.me_trial == 'U')].index)
            utils.pdCount(data,comment='Drop me_trial unknown',start=start)
        elif filter == 'non_cephallic':
            start = data.shape[0]
            data = data[(data['me_pres'] == 1 )]
            utils.pdCount(data,comment='Drop non-cephalic fetal presentations',start=start)
        elif filter == 'non_singletons':
            start = data.shape[0]
            data = data[(data['dplural'] == 1 )]
            utils.pdCount(data,comment='Drop non-singleton births',start=start)
        elif filter == 'first_birth':
            start = data.shape[0]
            data = data[(data['lbo'] == 1 )]
            utils.pdCount(data,comment='Restrict to nulliparous births',start=start)
        elif filter == 'unknown_wtgain_rec':
            start = data.shape[0]
            data = data.drop(data[(data.wtgain_rec > 8.0)].index)
            utils.pdCount(data,comment='Drop unknown wtgain_rec',start=start)
        elif filter == 'unknown_rf_diab':
            # consider only known RF_DIAB risk factor (values = Y,N)
            start = data.shape[0]
            data = data[(data['rf_diab'] == 'Y') | (data['rf_diab'] == 'N')]
            utils.pdCount(data,comment='Drop unknown rf_diab',start=start)
        elif filter == 'unknown_urf_diab':
            # consider only known urf_diab risk factor (1=yes,2=no)
            start = data.shape[0]
            if 'urf_diab' in data.columns:
                data = data[(data['urf_diab'] == 1) | (data['urf_diab'] == 2)]
                utils.pdCount(data,comment='Drop unknown urf_diab')
        elif filter == 'unknown_birth_order':
            start = data.shape[0]
            data = data[(data['lbo'] != 9)]
            data = data[(data['tbo'] != 9)]
            utils.pdCount(data,comment='Drop unknown live/total birth order',start=start)
        elif filter == 'unknown_previous_cesareans':
            start = data.shape[0]
            data['rf_cesarn'] = data['rf_cesarn'].astype(int)
            data = data.drop(data[(data.rf_cesarn == 99)].index)
            utils.pdCount(data,comment='Drop if previous Cesarean count is unknown',start=start)
        elif filter == 'unknown_bmi':
            start = data.shape[0]
            data['bmi_r'] = data['bmi_r'].astype(int)
            data.drop(data[(data.bmi_r == 9)].index,inplace=True)
            utils.pdCount(data,comment='Drop unknown BMI',start=start)
        elif filter == 'unknown_pwgt_r':
            start = data.shape[0]
            data['pwgt_r'] = data['pwgt_r'].astype(int)
            data.drop(data[(data.pwgt_r == 999)].index,inplace=True)
            utils.pdCount(data,comment='Drop unknown pre-pregnancy weight',start=start)
        elif filter == 'empty_mbrace':
            start = data.shape[0]
            data.drop(data[(data.mbrace == ' ')].index,inplace=True)
            data['mbrace'] = data['mbrace'].astype(int)
            utils.pdCount(data,comment='Drop empty mbrace',start=start)
        elif filter == 'unknown_or_unreported_smoking':
            start = data.shape[0]
            data['cig_0'] = data['cig_0'].astype(int)
            data['cig_1'] = data['cig_1'].astype(int)
            data['cig_2'] = data['cig_2'].astype(int)
            data['cig_3'] = data['cig_3'].astype(int)

            data.drop(data[(data.cig_0 == -999)].index,inplace=True)
            data.drop(data[(data.cig_1 == -999)].index,inplace=True)
            data.drop(data[(data.cig_2 == -999)].index,inplace=True)
            data.drop(data[(data.cig_3 == -999)].index,inplace=True)

            data.drop(data[(data.cig_0 == 99)].index,inplace=True)
            data.drop(data[(data.cig_1 == 99)].index,inplace=True)
            data.drop(data[(data.cig_2 == 99)].index,inplace=True)
            data.drop(data[(data.cig_3 == 99)].index,inplace=True)

            utils.pdCount(data,comment='Drop unknown smoking',start=start)
        else:
            utils.ERROR("Error: unknown data filter requested -> %s" % filter)
            
    print("Data filtering complete.")
    return(data)

#--
# convert data with "Y" and "N" designations to binary 1/0
def convert_yes_no_vars(data,varName,drop=True):

    # skip if bool variant is already present
    varName_bool = varName + '_bool'
    if varName_bool in data:
        return

    if varName in data:
        vals = sorted(data[varName].unique())
        if vals != ['N', 'Y']:
            utils.ERROR("Error: unexpected boolean value encountered for var -> %s " % varName)
        else:
            data[varName_bool] = data[varName].map(   {'Y':1, 'N':0})
            if drop:
                data.drop(columns=varName,inplace=True)
    else:
        utils.ERROR("Error: %s not present in dataframe (convert_yes_no_vars)" % varName)

