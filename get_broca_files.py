import shutil
import numpy as np
import pandas as pd
import sys


if __name__ == "__main__":

    input_file = 'data/english-results-data.csv'
    df = pd.read_csv(input_file, sep=';')
    broca_df = df.loc[df['WAB Type'] == 'Broca'].iloc[:,[0, 3]]

    print(broca_df.shape, file=sys.stderr)
    #print(broca_df)

    for _, file in broca_df.iterrows():

        if file['Participant ID'].startswith('NEURAL'):
            shutil.copy(f"data/Aphasia/all_files/{file['Participant ID'][6:]}.cha", f"data/Aphasia/broca_files/")
        elif file['Participant ID'].startswith('UMD-'):
            try:
                shutil.copy(f"data/Aphasia/all_files/{file['Participant ID'][4:]}.cha", f"data/Aphasia/broca_files/")
            except FileNotFoundError:
                print(f"This file: (UMD-){file['Participant ID'][-6:]}.cha is not present in the downloaded data",
                      file=sys.stderr)
        elif file['Participant ID'].startswith('fridriksson-'):
            try:
                shutil.copy(f"data/Aphasia/all_files/{file['Participant ID'][-6:]}.cha", f"data/Aphasia/broca_files/")
            except FileNotFoundError:
                print(
                    f"This file: (fridriksson-[1|2]){file['Participant ID'][-6:]}.cha is not present in the downloaded data",
                    file=sys.stderr)
        else:
            shutil.copy(f"data/Aphasia/all_files/{file['Participant ID']}.cha", f"data/Aphasia/broca_files/")
