import os
import csv
from datetime import datetime
from logger import get_logger

logger = get_logger(__name__)

ATTENDANCE_FILE = "data/attendance.csv"


def ensure_attendance_file():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "date", "punch_in", "punch_out"])
        logger.info("Attendance file created")


def mark_attendance(user_id):
    ensure_attendance_file()

    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    records = []

    with open(ATTENDANCE_FILE, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    # Check existing record
    for row in records:
        if row["user_id"] == user_id and row["date"] == today:
            if row["punch_out"] == "":
                row["punch_out"] = current_time
                logger.info(f"Punch-out marked for {user_id} at {current_time}")
            else:
                logger.warning(f"Duplicate attendance attempt by {user_id}")
            break
    else:
        # No record found → punch-in
        records.append(
            {
                "user_id": user_id,
                "date": today,
                "punch_in": current_time,
                "punch_out": "",
            }
        )
        logger.info(f"Punch-in marked for {user_id} at {current_time}")

    # Write back to CSV
    with open(ATTENDANCE_FILE, mode="w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["user_id", "date", "punch_in", "punch_out"]
        )
        writer.writeheader()
        writer.writerows(records)


# -----------------------------
# Standalone Test
# -----------------------------
if __name__ == "__main__":
    uid = input("Enter user ID: ").strip()
    mark_attendance(uid)
