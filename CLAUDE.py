"""
Zerodha Access Token Generator
Install required package: pip install kiteconnect
"""

from kiteconnect import KiteConnect

# Step 1: Enter your credentials
API_KEY = "wi56icsdclxzrnfy"

API_SECRET = "ikkz08layudgxkhp8nsdizyv2szjyix1"


# Step 2: Initialize KiteConnect
kite = KiteConnect(api_key=API_KEY)

# Step 3: Generate login URL
print("=" * 60)
print("STEP 1: Open this URL in your browser and login:")
print("=" * 60)
print(kite.login_url())
print()

# Step 4: Get request token from user
print("=" * 60)
print("STEP 2: After login, copy the 'request_token' from URL")
print("=" * 60)
print("The URL will look like:")
print("https://kite.trade/?request_token=ABC123&action=login&status=success")
print()
request_token = input("Paste your request_token here: ").strip()

# Step 5: Generate session and get access token
try:
    data = kite.generate_session(request_token, api_secret=API_SECRET)
    access_token = data["access_token"]

    print()
    print("=" * 60)
    print("SUCCESS! Your Access Token:")
    print("=" * 60)
    print(access_token)
    print()
    print("⚠️  Save this token - it expires at midnight!")
    print("=" * 60)

    # Optional: Set access token and test
    kite.set_access_token(access_token)
    profile = kite.profile()
    print(f"\n✅ Verified! Logged in as: {profile['user_name']} ({profile['email']})")

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nCommon issues:")
    print("- Check if your API Key and Secret are correct")
    print("- Make sure request_token is copied completely")
    print("- Request tokens expire quickly - generate a new one if needed")