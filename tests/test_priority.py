from app.services import priority, routing


def test_calm_question_is_low():
    p, reasons = priority.predict("Hi, could you tell me the expected delivery time for Pune?")
    assert p == "low"


def test_delay_with_repeat_contact_is_high():
    text = "I ordered a product five days ago, but it still hasn't been delivered. I have contacted support twice and received no update."
    p, reasons = priority.predict(text, category="delivery", sentiment="frustrated")
    assert p == "high"
    assert any("repeated contact" in r for r in reasons)


def test_locked_account_is_at_least_medium():
    p, _ = priority.predict("I am locked out of my account and the OTP never arrives", category="account")
    assert p in ("medium", "high")


def test_hindi_delay_with_repeat_contact_is_high():
    text = "मैंने 6 दिन पहले ऑर्डर किया था, अभी तक डिलीवर नहीं हुआ। मैंने दो बार सपोर्ट से बात की लेकिन कोई जवाब नहीं मिला।"
    p, reasons = priority.predict(text, category="delivery", sentiment="frustrated")
    assert p == "high"
    assert any("repeated contact" in r for r in reasons)
    assert any("waiting 6 days" in r for r in reasons)


def test_hinglish_locked_account_urgent_is_high():
    text = "Bhai mera account lock ho gaya hai, OTP bhi nahi aa raha 4 baar try kiya. Aaj hi order karna hai, jaldi help karo."
    p, _ = priority.predict(text, category="account", sentiment="frustrated")
    assert p == "high"


def test_hindi_numerals_count_as_days():
    p, reasons = priority.predict("१२ दिन से पार्सल नहीं आया", category="delivery")
    assert any("waiting 12 days" in r for r in reasons)


def test_routing_known_and_unknown():
    assert routing.route("delivery") == "Logistics Support"
    assert routing.route(None) == routing.DEFAULT_TEAM
