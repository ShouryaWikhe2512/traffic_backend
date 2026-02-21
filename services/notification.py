from twilio.rest import Client
import time
import logging

class NotificationService:
    def __init__(self, account_sid, auth_token, from_number):
        self.client = Client(account_sid, auth_token)
        self.from_number = from_number
        self.police_contacts = {
            "NORTH": "918956747898",
            "SOUTH": "918956747898",
            "EAST": "919356431340",
            "WEST": "919356431340"
        }
        self.last_call_time = {} # To prevent spamming
        self.cooldown = 45 # Balanced cooldown

    def _format_number(self, num: str) -> str:
        """Ensures number is in E.164 format (+91...)"""
        clean = "".join(filter(str.isdigit, num))
        if len(clean) == 10:
            return f"+91{clean}"
        if len(clean) == 12 and clean.startswith("91"):
            return f"+{clean}"
        if not clean.startswith("+") and len(clean) > 5:
            return f"+{clean}"
        return num

    def notify_emergency(self, direction: str, vehicle_type: str):
        direction = direction.upper()
        current_time = time.time()
        
        # Debounce/Cooldown check
        if direction in self.last_call_time:
            if current_time - self.last_call_time[direction] < self.cooldown:
                print(f"[TWILIO] Call to {direction} suppressed (Cooldown: {int(self.cooldown - (current_time - self.last_call_time[direction]))}s left)")
                return

        raw_number = self.police_contacts.get(direction)
        if not raw_number:
            print(f"[TWILIO] Error: No contact found for {direction} Approach")
            return

        target_number = self._format_number(raw_number)

        try:
            print(f"[TWILIO] ATTEMPTING CALL -> {direction} | Target: {target_number} | Vehicle: {vehicle_type}")
            
            # Create TwiML for the voice call
            twiml = f"<Response><Say voice='alice'>Emergency Alert. An {vehicle_type.replace('_', ' ')} has been detected at the {direction} approach. Priority preemption is active.</Say></Response>"
            
            call = self.client.calls.create(
                to=target_number,
                from_=self.from_number,
                twiml=twiml
            )
            
            self.last_call_time[direction] = current_time
            print(f"[TWILIO] SUCCESS | SID: {call.sid} | Status: {call.status}")
            
        except Exception as e:
            print(f"[TWILIO] CRITICAL FAILURE: {str(e)}")
            if "verified" in str(e).lower():
                print("[TWILIO] TIP: You are likely using a Trial Account. You must VERIFY the target phone number in your Twilio Console first.")
