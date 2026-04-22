def mock_lead_capture(name: str, email: str, platform: str) -> str:
	"""
	Mock CRM API call. In production this would POST to a backend service.
	Called only after all three lead fields are present.
	"""
	print("\n" + "=" * 50)
	print("[TOOL CALLED] mock_lead_capture()")
	print(f"  -> Name:     {name}")
	print(f"  -> Email:    {email}")
	print(f"  -> Platform: {platform}")
	print(f"Lead captured successfully: {name}, {email}, {platform}")
	print("=" * 50 + "\n")

	return f"Lead captured successfully for {name} ({email}) on {platform}."
