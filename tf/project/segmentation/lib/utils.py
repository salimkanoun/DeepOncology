def secsToHms(secs):
    hours = secs // 3600
    secs -= hours * 3600
    mins = secs // 60
    secs -= mins * 60
    return hours, mins, secs
    
def sec2str(seconds):
    return "%02d:%02d:%02d" % secsToHms(seconds)