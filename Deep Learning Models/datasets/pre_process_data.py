from datasets.utils import *
import os.path
import pandas as pd
import argparse
from scripts.process_english import process_english
from scripts.process_apply import process_apply
from scripts.process_grades import process_grades
from scripts.process_honour1 import process_honour
from scripts.process_project import process_project
from scripts.process_ranking import process_ranking
from scripts.process_unversitylevel import process_university
from scripts.resume_process import resume_process


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='datasets', help='root')
opt = parser.parse_args()

stu_file = os.path.join(opt.data_root, 'stu_base.csv')
honour_file = os.path.join(opt.data_root, 'stu_honour.csv')
project_file = os.path.join(opt.data_root, 'stu_project.csv')
university_file = os.path.join(opt.data_root, 'db_university_free.csv')


process_english(stu_file)
process_apply(stu_file)
process_grades(stu_file)
process_honour(stu_file, honour_file)
process_project(stu_file, project_file)
process_ranking(stu_file)
process_university(stu_file, university_file)
# resume_process(stu_file)
