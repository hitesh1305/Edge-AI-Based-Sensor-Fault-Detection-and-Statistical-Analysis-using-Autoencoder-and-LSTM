from pathlib import Path

p = Path("main.py")
text = p.read_text(encoding="utf-8")

marker = '    window_generator = WindowGenerator('
export = '''    # Export the exact processed test dataframe for Raspberry Pi validation.
    from pathlib import Path
    pi_export = Path("output/tables/processed_test_df_for_pi.csv")
    pi_export.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(pi_export, index=False)
    print(f"Pi validation export: {pi_export.resolve()} | shape={test_df.shape}")

'''

if "processed_test_df_for_pi.csv" not in text:
    if marker not in text:
        raise SystemExit("STOP: Could not locate window-generator insertion point.")
    text = text.replace(marker, export + marker, 1)
    p.write_text(text, encoding="utf-8")
    print("Export step inserted into main.py.")
else:
    print("Export step already exists; no duplicate inserted.")
