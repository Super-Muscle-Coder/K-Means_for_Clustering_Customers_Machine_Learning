import pandas as pd

def check_full_row_duplicates(csv_path):
    """
    Kiểm tra và in ra tất cả các dòng trùng lặp hoàn toàn trong file CSV (không còn cột ID).
    Nếu có trùng lặp, in ra toàn bộ các dòng trùng lặp.
    """
    print("="*100)
    print("KIỂM TRA TRÙNG LẶP TOÀN BỘ DÒNG\n")

    # Đọc dữ liệu
    df = pd.read_csv(csv_path, sep="\t")

    # Loại bỏ cột ID nếu tồn tại
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
        print("Đã loại bỏ cột ID.\n")

    # Tìm các dòng trùng lặp hoàn toàn (giữ lại tất cả các dòng trùng lặp)
    duplicated_mask = df.duplicated(keep=False)
    duplicated_rows = df[duplicated_mask]

    num_duplicates = df.duplicated(keep='first').sum()
    num_groups = duplicated_rows.groupby(list(df.columns)).ngroups if not duplicated_rows.empty else 0

    if duplicated_rows.empty:
        print("Không có dòng trùng lặp hoàn toàn trong dataset.")
    else:
        print(f"Phát hiện {num_duplicates} dòng trùng lặp hoàn toàn (full-row duplicates).")
        print(f"Số nhóm trùng lặp: {num_groups}")
        print("\nTẤT CẢ CÁC DÒNG TRÙNG LẶP:")
        # Sắp xếp để các dòng giống nhau nằm cạnh nhau
        duplicated_rows_sorted = duplicated_rows.sort_values(list(df.columns)).reset_index(drop=True)
        print(duplicated_rows_sorted.to_string(index=False))
    print("="*100)


if __name__ == "__main__":
    # Đường dẫn tới file CSV đã làm sạch (không còn cột ID)
    csv_path = r"C:\Project\Machine_Learning\Machine_Learning\Dataset\Customer_Behavior.csv"
    check_full_row_duplicates(csv_path)