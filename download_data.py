import gdown

SIM_INSERTION_HUMAN = "1RgyD0JgTX30H4IM5XZn8I3zSV_mr8pyF"
SIM_INSERTION_SCRIPTED = "1TsojQQSXtHEoGnqgJ3gmpPQR2DPLtS2N"
SIM_TRANSFER_CUBE_HUMAN = "1sc-E4QYW7A0o23m1u2VWNGVq5smAsfCo"
SIM_TRANSFER_CUBE_SCRIPTED = "1aRyoOhQwxhyt1J8XgEig4s6kzaw__LXj"
folders = [SIM_INSERTION_HUMAN, SIM_INSERTION_SCRIPTED, SIM_TRANSFER_CUBE_HUMAN, SIM_TRANSFER_CUBE_SCRIPTED]
fourtyeight = ["18Cudl6nikDtgRolea7je8iF_gGKzynOP", "1wfMSZ24oOh5KR_0aaP3Cnu_c4ZCveduB", "18smMymtr8tIxaNUQ61gW6dG50pt3MvGq", "1pnGIOd-E4-rhz2P3VxpknMKRZCoKt6eI"]
fourtynine = ["1C1kZYyROzs-PrLc0SkDgUgMi4-L3lauE", "17EuCUWS6uCCr6yyNzpXdcdE-_TTNCKtf", "1Nk7l53d9sJoGDBKAOnNrExX5nLacATc6", "1GKReZHrXU73NMiC5zKCq_UtqPVtYq8eo"]

# Download folders
for i, folder in enumerate(folders):
    if i == 0:
        name = "sim_insertion_human"
    elif i == 1:
        name = "sim_insertion_scripted"
    elif i == 2:
        name = "sim_transfer_cube_human"
    else:
        name = "sim_transfer_cube_scripted"
    gdown.download_folder(id=folder, output=f"data/{name}", remaining_ok=True)

# Download remaining files
for i, files in enumerate(zip(fourtyeight, fourtynine)):
    if i == 0:
        name = "sim_insertion_human"
    elif i == 1:
        name = "sim_insertion_scripted"
    elif i == 2:
        name = "sim_transfer_cube_human"
    else:
        name = "sim_transfer_cube_scripted"

    for file in files:
        gdown.download(id=file, output=f"data/{name}/")