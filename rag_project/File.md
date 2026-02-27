### Sec Filling

## Testing
- test_analyze_sec_document.py use for analyze gpt prompt to indetify which type of document.



### GUNSHOT

## Use In development


# This file is use for find the high value followers tweets and anylize with gpt and get score of the tweet regarding anti trust..
- high_value_followers_tweet_analyzer.py


# This file is use for run the gunshot approach step by step like
* Fetch all the follower of the company.
* Anylize the high value followers
* Fetch tweet of high value followers related to company or products.
- high_value_followers_orchestrator.py

# This file is used for find the high value followers has no bio or empty bio with some other min filter , then we search tweets of that user with tweeter query and then if any single tweet we found then consider follower as hig value follower.(OneTime run)
- high_value_followers_tweet_search.py

# This file is used for find the high value followers has Bio with some other min filter. we analyz the each user bio with GPT and get score and save the follower have score >= 6. (OneTime run)
- high_value_followers_processor.py



## Debugg

# Below root level file is use for directly test high_value_followers_processor.py file functionality
- test_high_value_followers.py

# Below root level file is use for direcly test  high_value_followers_tweet_search.py file functionality.
- test_tweet_search_high_value.py

# From all the followers filterout the follower base on given parameter like min post, followers.then build the tweetwer api search query and call the Api to fetch user by user tweet.
# This approach is only for testing how many tweet we get by searching the all the tweet.
# 09/09/2025 still not confirm, which flow have to take.
- high_value_followers_tweet_search_test.py

# Filter 1000 tweet of the user tweet and the output file we use in tweet_scorer.py file as input.
# Juniper_Networks_merged_tweets_20250904_145830.json are the sample file for Juniper company.
- filter_tweets_script.py


# Read The user tweets josn file  and ask gpt to score them.
- python tweet_scorer.py input_file.json output_file.json

# Tweet Search with from:username and products.This is we do in Gunset Approach after getting high value follower we fetch all tweet of the follower regarding product or company.
- python tweet_search_simple.py 