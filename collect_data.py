import pandas
import datetime
import time
import calendar
from pmaw import PushshiftAPI
import praw

def customFilter(comment):
    return comment['body'] != '[deleted]' and comment['body'] != '[removed]'

def getInfo(comment):
    result = {
        'body': comment['body'],
        'controversiality': comment['controversiality'],
        'permalink': comment['permalink'],
        'score': comment['score'],
        'subreddit': comment['subreddit'],
        'id' : comment['id'],
        'timestamp' : comment['created_utc']
    }
    return result

def getEpochTime(year, month):
    return calendar.timegm(datetime.datetime(year, month, 1, 0, 0, 0).timetuple())

def epochToString(epochTime):
    return datetime.datetime.fromtimestamp(epochTime).strftime('%c')

iteration = 0
ids = {}
commentList = []
api = PushshiftAPI()
startYear = 2020
startMonth = 1
startTime = getEpochTime(startYear, startMonth) #GMT 2019-01-01 epoch time
endTime = getEpochTime(startYear, startMonth + 1)
subreddit = "Liberal"
limit = 100000

# monitor time it took to collect the data
scriptStartTime = time.time()

# len(commentList) < limit

while startYear < 2021:
    iteration += 1
    print(f'Current comment count: {len(commentList)}\n Current iteration: {iteration}')
    print(f'Collecting comments from {epochToString(startTime)} to {epochToString(endTime)}')
    # comments is a generator object
    comments = api.search_comments(subreddit = subreddit, size=25000, since = startTime, until = endTime, mem_safe = True)
    print(f'Collected {len(comments)} comments \n')

    for comment in comments:
        if comment['id'] in ids:
            continue
        else:
            if customFilter(comment):
                commentList.append(getInfo(comment))
                ids[comment['id']] = 1

    # advance search time to limit duplicates
    startMonth += 1
    if startMonth == 12 :
        startTime = getEpochTime(startYear, startMonth)
        endTime = getEpochTime(startYear + 1, 1)
    elif startMonth == 13 :
        startYear += 1
        startMonth = 1
        startTime = getEpochTime(startYear, startMonth)
        endTime = getEpochTime(startYear, startMonth + 1)
    else:
        startTime = getEpochTime(startYear, startMonth)
        endTime = getEpochTime(startYear, startMonth + 1)
    print('sleeping...\n')
    time.sleep(10)
    # prevent inifinite loop if there aren't enough unique comments
    if iteration > 2500 :
        break

print(f'Retrieved {len(commentList)} comments \n')
print(f'task took {(time.time() - scriptStartTime) / 60} minutes')
    
df = pandas.DataFrame(commentList)
#df.head(5)

df.to_csv('./lib_comments.csv', header = True, index = False, columns = list(df.axes[1]))
