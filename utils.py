'''
Return number of minutes and seconds between given start and end time
Params:
    start_time: i.e Time.time()
    end_time: i.e Time.time()
Returns:
    minutes
    seconds
'''
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
