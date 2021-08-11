import pandas as pd

df = pd.read_csv("CSV output/preprocessed_bearings.csv")
print(df.shape)
df = df[df["status"] == 0]
df = df.drop(['rpm_mean', 'w_mean','w_range','w_max','status'], axis=1)
print(df.shape)
print(df.columns)


# Difference between a1 and a2 into columns.
columns_to_get_difference = (('a1_x_mean','a2_x_mean'),('a1_y_mean','a2_y_mean'),('a1_z_mean','a2_z_mean'),
                             ('a1_x_range','a2_x_range'),('a1_y_range','a2_y_range'),('a1_z_range','a2_z_range'),
                             ('a1_x_min','a2_x_min'),('a1_y_min','a2_y_min'),('a1_z_min','a2_z_min'),
                             ('a1_x_max','a2_x_max'),('a1_y_max','a2_y_max'),('a1_z_max','a2_z_max'),
                             ('a1_x_fft_mean','a2_x_fft_mean'),('a1_y_fft_mean','a2_y_fft_mean'),
                             ('a1_z_fft_mean','a2_z_fft_mean'),('a1_x_ff_range','a2_x_fft_range'),
                             ('a1_y_fft_range','a2_y_fft_range'),('a1_z_fft_range','a2_z_fft_range'),
                             ('a1_x_fft_min','a2_x_fft_min'),('a1_y_fft_min','a2_y_fft_min'),
                             ('a1_z_fft_min','a2_z_fft_min'),('a1_x_fft_max','a2_x_fft_max'),
                             ('a1_y_fft_max','a2_y_fft_max'),('a1_z_fft_max','a2_z_fft_max'))

def dataframe_with_differences_between_a1_and_a2_with_original_columns_deleted(df: pd.DataFrame, columns_to_compare_and_delete: tuple[tuple[str,str]]) -> pd.DataFrame:
    """
    Calculates the difference between a1 and a2 bearings, adds the result as a new column and deletes the a1-column.
    Renames every 'a2_' column so it no longer contains 'a2_'.

    :return Altered pd.DataFrame
    """
    for columns in columns_to_compare_and_delete:
        a1_column = columns[0]
        a2_column = columns[1]
        difference = df[a2_column] - df[a1_column]
        df = df.drop([a1_column], axis=1)
        new_column_name = a2_column[3:] + "_difference"
        df.rename(columns={a2_column: a2_column[3:]}, inplace=True)
        df[new_column_name] = difference
    return df

df = dataframe_with_differences_between_a1_and_a2_with_original_columns_deleted(df, columns_to_get_difference)
print(df.shape)
df = df.drop([x for x in df.columns if x.endswith("_difference")], axis=1)
print(df.shape)
df.to_csv("CSV output/clustering_ready", index=False)