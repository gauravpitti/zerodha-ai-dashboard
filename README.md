first open pycharm or any other app like visual code i preffer pycharm 
in pycharm open claude.py file 
run claude.py in terminal by typing streamlit run claude.py 
but before running chanege the api key and api secret in the code 
to get api key and api secret login to https://developers.kite.trade
and create a new app enter the details like app name,zerodha client id, app icon and for redirect url http://127.0.0.1:5000 and for post back url https://kite.trade
then clcik create app and you app will be created from that copy your api key and api secret and paste it in claude.py where api key and api secret is written on line 9 and 11
then run it with streamlit run claude.py it will automatically redirect you to the zerodha login page once logged in copy the request token from the url like 
https://kite.trade/?request_token=ABC123&action=login&status=success
only copy the right part thats ABC123 not full &action
and paste that token in your pycharm terminal you will get your acces token this will expire every midnight so if you want to then you have to create it daily to login following same procedure 
after getting all this run the zerodhault.py in terminal by typing streamlit run zerodhault.py and you will automatically be redirected to our dashboard
and then enter your api key and access token
to enable ai chat and insights login to https://aistudio.google.com/app/api-keys and also the link is on dashboard 
from there you will get your ai api key enter that key and you are all set
