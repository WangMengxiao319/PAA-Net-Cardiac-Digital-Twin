import os
import numpy as np
import pymupdf
import matplotlib.pyplot as plt

# Settings

PDF_FOLDER = r"/path/to/dataset/ECG_series"
OUT_FOLDER = r"/path/to/output/ECG_series_digitalized_full"
os.makedirs(OUT_FOLDER, exist_ok=True)

LEAD_ORDER = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3",
    "V4", "V5", "V6"
]

CURVE_OFFSET = 3          # curve 4 → I, curve 5 → II, ...
CALIB_CURVE_INDEX = 0     # calibration pulse curve index
MEAIN_LEAD_INDEX = 3 + 12 + 3     # lead II curve index (full 10s)
TARGET_SAMPLES = 5000
CALIB_MV = 1            # calibration pulse = 1 mV

def TwoLineConnect(currentLine, nextLine):
    x1, y1 = currentLine[-1]
    x2, y2 = nextLine[0]
    return abs(x1 - x2) + abs(y1 - y2) < 1e-3
def extract_all_curves(vector_graph, ptm, color):
    lines = []
    for element in vector_graph:
        if element.get("color") == color:
            for item in element.get("items", []):
                if item[0] == "l":
                    p1, p2 = item[1] * ~ptm, item[2] * ~ptm
                    lines.append((p1, p2))

    if not lines:
        return np.array([]), np.array([])

    points = [[(l[0].x, l[0].y), (l[1].x, l[1].y)] for l in lines]

    curves = []
    current = []
    for i in range(len(points) - 1):
        currentLine = points[i]
        nextLine = points[i + 1]
        current.append([currentLine[0], currentLine[1]])

        if TwoLineConnect(currentLine, nextLine) and (i < len(points) - 2):
            continue
        else:
            curves.append(current)
            current = []

    if len(points) == 1:
        curves.append([[points[0][0], points[0][1]]])
        
    return curves

def extract_single_curve(curves, curve_index):
    
    if curve_index >= len(curves):
        return np.array([]), np.array([])

    chosen = curves[curve_index]

    xs, ys = [], []
    last = None
    for (a, b) in chosen:
        if (last is None) or (not (np.isclose(a[0], last[0]) and np.isclose(a[1], last[1]))):
            xs.append(a[0])
            ys.append(a[1])
        xs.append(b[0])
        ys.append(b[1])
        last = b

    return np.array(xs), np.array(ys)


def robust_height(arr):
    lo = np.percentile(arr, 1)
    hi = np.percentile(arr, 99)
    return hi - lo

# ----------------- Revised main processing workflow -----------------
if __name__ == "__main__":
    for folder in os.listdir(PDF_FOLDER):
        os.makedirs(os.path.join(OUT_FOLDER, folder), exist_ok=True)
        pdf_files = [f for f in os.listdir(os.path.join(PDF_FOLDER, folder)) if f.lower().endswith(".pdf")]
        
        for pdf_name in pdf_files:
            pdf_path = os.path.join(PDF_FOLDER, folder, pdf_name)
            patient_id = os.path.splitext(pdf_name)[0]
            out_csv = os.path.join(OUT_FOLDER, folder, f"{patient_id}.csv")
            
            if os.path.exists(out_csv):
                print(f"\n{patient_id} already processed, skipping.")
                continue

            print(f"\nProcessing {patient_id} ...")
            doc = pymupdf.open(pdf_path)
            page = doc[0]
            ptm = page.transformation_matrix
            vector_graph = page.get_drawings()
            color = (0.0, 0.0, 0.0)

            curves = extract_all_curves(vector_graph, ptm, color)

            # 1. Extract the calibration voltage.
            x_c, y_c = extract_single_curve(curves, CALIB_CURVE_INDEX)
            if len(x_c) == 0:
                print("Calibration pulse not found — skipping.")
                doc.close()
                continue

            x_c_rot = y_c
            y_c_rot = -x_c
            calib_height_units = robust_height(y_c_rot)
            if calib_height_units <= 0:
                print("Invalid calibration height — skipping.")
                doc.close()
                continue

            UNITS_TO_MV = CALIB_MV / calib_height_units
            print(f"  Calibration scale: {UNITS_TO_MV:.6f} mV/unit")

            # ----------------- Core modification -----------------
            # 2. Extract the dominant long lead II first to establish the physical X range of the global 10-second timeline.
            x_main_raw, y_main_raw = extract_single_curve(curves, MEAIN_LEAD_INDEX)
            if len(x_main_raw) == 0:
                print("  Main lead (II) not found — skipping.")
                doc.close()
                continue
                
            x_main_rot = y_main_raw  
            X_MIN = x_main_rot.min()
            X_MAX = x_main_rot.max()

            # 3. Extract the 12 leads in order and map them to the global timeline.
            lead_signals = []
            t_new = np.linspace(0, 1, TARGET_SAMPLES)

            for i in range(12):
                if i == 1:
                    curve_idx = MEAIN_LEAD_INDEX
                else:
                    curve_idx = CURVE_OFFSET + i
                    
                x_raw, y_raw = extract_single_curve(curves, curve_idx)

                if len(x_raw) == 0:
                    print(f"  Missing curve {curve_idx}, filling zeros.")
                    lead_signals.append(np.zeros(TARGET_SAMPLES))
                    continue

                x_rot = y_raw
                y_rot = -x_raw

                y_mv = (y_rot - np.mean(y_rot)) * UNITS_TO_MV

                t = (x_rot - X_MIN) / (X_MAX - X_MIN)
                

                sort_idx = np.argsort(t)
                t_sorted = t[sort_idx]
                y_mv_sorted = y_mv[sort_idx]

                y_res = np.interp(t_new, t_sorted, y_mv_sorted, left=0.0, right=0.0)

                lead_signals.append(y_res)

            lead_signals = np.array(lead_signals)  # 12 × 5000
            # ------------------------------------------------

            # Save the CSV file.
            np.savetxt(out_csv, lead_signals, delimiter=",", fmt="%.4f")
            print(f"  Saved {out_csv}")

            # Plot for verification.
            plt.figure(figsize=(12, 10))
            for i, lead_name in enumerate(LEAD_ORDER):
                plt.subplot(12, 1, i + 1)
                plt.plot(t_new, lead_signals[i], linewidth=0.8)
                plt.ylabel(lead_name, rotation=0, labelpad=15)
                plt.xticks([])
                # Fix the y-axis range to make the alignment clearer.
                plt.ylim(-2, 2) 

            plt.suptitle(f"ECG (mV) – {patient_id}")
            plt.xlabel("Normalized Time (0-10s)")
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(os.path.join(OUT_FOLDER, folder, f"{patient_id}.png"), dpi=150)
            plt.close()

            doc.close()

    print("\nBatch processing completed.")
