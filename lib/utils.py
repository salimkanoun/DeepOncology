import os
import re


def read_cfg(filepath):
    var = dict()
    exec(open(filepath).read(), var)

    return var


def secsToHms(secs):
    hours = secs // 3600
    secs -= hours * 3600
    mins = secs // 60
    secs -= mins * 60
    return hours, mins, secs


def sec2str(seconds):
    return "%02d:%02d:%02d" % secsToHms(seconds)


def display_info(img):
    print('img information :')
    print('\t Origin    :', img.GetOrigin())
    print('\t Size      :', img.GetSize())
    print('\t Spacing   :', img.GetSpacing())
    print('\t Direction :', img.GetDirection())


def get_study_uid(img_path):
    return re.sub('_nifti_(PT|mask|CT)\.nii(\.gz)?', '', os.path.basename(img_path))
