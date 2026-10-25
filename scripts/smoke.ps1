param([string]$BaseUrl = "http://localhost:8000")
if (-not $env:QA_API_TOKEN) { throw "Set QA_API_TOKEN in this terminal first." }
curl.exe -i "$BaseUrl/health"
curl.exe -i "$BaseUrl/info"
curl.exe -i -H "Authorization: Bearer wrong" "$BaseUrl/info"
curl.exe -i -H "Authorization: Bearer $env:QA_API_TOKEN" "$BaseUrl/info"
curl.exe -i -X POST "$BaseUrl/predict" -H "Authorization: Bearer $env:QA_API_TOKEN" -H "Content-Type: application/json" -d "@request.json"
