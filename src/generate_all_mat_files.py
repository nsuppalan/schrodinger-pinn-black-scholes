# generate_all_mat_files.py

from generate_option_mat import create_options_mat
from generate_equity_mat import create_equity_mat

# List of expiry dates to process
expiries = ["20250529"]

for expiry in expiries:
    print(f"\n📅 Generating .mat files for expiry {expiry}...")

    # Define filenames dynamically
    opt_file = f"OPT_RELIANCE_{expiry}.csv"
    eqt_file = f"EQT_RELIANCE_{expiry}.csv"
    opt_mat_output = f"OPT_PINN_TRAIN_DATA_{expiry}.mat"
    eqt_mat_output = f"EQT_PINN_TRAIN_DATA_{expiry}.mat"
    '''
    # Generate options .mat file (linked with corresponding equity file for spot alignment)
    create_options_mat(
        opt_file=opt_file,
        expiry=expiry,
        output_mat_file=opt_mat_output,
        eqt_file=eqt_file
    )
    '''

    # create_equity_mat only supports positional args — so call it accordingly
    create_equity_mat(
        eqt_file,
        expiry,
        eqt_mat_output
    )

print("\n All .mat files successfully generated for all expiries.")
