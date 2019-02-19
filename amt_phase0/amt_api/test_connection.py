MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'

mturk = boto3.client('mturk',
   aws_access_key_id = "AKIAIXEDNTD6NT26FTUA",
   aws_secret_access_key = "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",
   region_name='us-east-1',
   endpoint_url = MTURK_SANDBOX
)

print("I have $" + mturk.get_account_balance()['AvailableBalance'] + " in my Sandbox account")
