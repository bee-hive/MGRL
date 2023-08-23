import pandas as pd

datadir = '../../files/mimiciv/0.4/'
cohort_csv = datadir + 'icu/icustays.csv'
patients_csv = datadir + 'core/patients.csv'
weight_csv = datadir + 'icu/inputevents.csv'
vitals_csv = datadir + 'icu/chartevents.csv'
labs_csv = datadir + 'ICU PTS COHORT LABS.csv'
meds_csv = datadir + 'ICU PTS COHORT MEDS.csv'
orders_csv = datadir + 'ICU PTS COHORT ORDERS.csv'
proc_csv = datadir + 'ICU PTS COHORT PROCEDURES.csv'
provider_csv = datadir + 'ICU PTS COHORT PROVIDERS.csv'
fin_csv = datadir + 'ICU PTS COHORT FIN CLASS.csv'
drug_csv = datadir + 'ICU PTS COHORT DRG.csv'
dx_csv = datadir + 'ICU PTS COHORT DX.csv'
outputdir = '../mimic_iv/'
checkpointdir = outputdir + 'checkpoints_hosp/'
policydir = '../policies/'

def read_eicu_csv(csv_path):
        #print("CSV Path: ", outputdir+csv_path)
        return pd.read_csv(outputdir+csv_path, encoding='raw_unicode_escape')