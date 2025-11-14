- to normally run build_schedule -> python3 build_schedule.py
- to check it against the checker python file -> python3 check_schedule.py build_schedule.py
- to generate two different variations of the schedule -> python3 generate_two_schedules.py word_pairs_E2.csv --outdir outputs
- to check these two -> python3 check_schedule.py outputs/output_schedule_1.csv
python3 check_schedule.py outputs/output_schedule_2.csv
