# Android external connectivity check

These are instructions for a separately authorized HTTPS session, not permission
to open a tunnel now. The historical filename is retained to preserve links.

1. On the Samsung phone, disable Wi-Fi in Settings and enable mobile data. Confirm
   4G/5G/LTE and internet access. Keep the laptop on independent Wi-Fi, never the
   phone hotspot. Record the user's confirmation of both networks.
2. Obtain the new authorized HTTPS URL. Open `/health` in Chrome; expect
   `{"status":"ok"}`. Open `/info` without a token; expect HTTP 401 and
   `{"error":"unauthorized"}`. A missing header is different from an invalid token.
3. In HTTP Request Shortcuts, create a GET request to `/info`. Add header
   `Authorization` with value `Bearer wrong`. Expect the same 401 error.
4. In Global Variables, create a **Password Input** variable named `api_token`.
   Set title to `Pimientero temporary token`. Disable **Remember value**, URL
   encoding and JSON encoding. Never paste the token into the request URL.
5. Edit the Authorization value to `Bearer {api_token}`, inserting the variable
   through the app's selector. Run the request and enter the temporary token only
   in the password dialog. Expect HTTP 200 and JSON containing team, model and
   params; verify checkpoint hash/steps against the actual selected model.
6. Create a POST request to `/predict`. Reuse Authorization and add
   `Content-Type: application/json`. Use this body:

```json
{"id":"phone-01","context":"Paris is the capital of France.","question":"What is the capital of France?"}
```

Expect id `phone-01`, answer `Paris`, start `0`, and a numeric score from 0 to 1.
The score can change between models. A null answer must have a null start. For
non-null answers, the context substring at the returned offset must match exactly.
7. Record HTTP status and exact JSON, without storing the password or screenshots
   containing it. Ask the laptop operator to close the tunnel after completion,
   verify the URL is inaccessible, and revoke the temporary token by stopping its
   server. Delete the temporary shortcut/variable from the borrowed phone.

Historical evidence: `logs/continued20k/mobile_connectivity.json` records five
successful Samsung mobile-data checks for continued20k. It is not evidence of a
new quality40k or pretrained-model test and contains no mobile latency benchmark.
