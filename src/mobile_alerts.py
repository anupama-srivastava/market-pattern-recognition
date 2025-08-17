"""
Mobile Alerts Module

This module implements mobile trading alerts using various notification services
including Discord, Slack, and email notifications for trading signals.
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dataclasses import dataclass
from enum import Enum

class AlertType(Enum):
    """Types of trading alerts"""
    PATTERN_DETECTED = "pattern_detected"
    EARNINGS_ALERT = "earnings_alert"
    OPTIONS_FLOW = "options_flow"
    PRICE_ALERT = "price_alert"
    VOLUME_SPIKE = "volume_spike"

@dataclass
class AlertMessage:
    """Alert message structure"""
    symbol: str
    alert_type: AlertType
    message: str
    timestamp: datetime
    metadata: Dict

class MobileAlertSystem:
    """Mobile trading alerts implementation"""
    
    def __init__(self, discord_webhook: Optional[str] = None, slack_webhook: Optional[str] = None, email_config: Optional[Dict] = None):
        self.discord_webhook = discord_webhook
        self.slack_webhook = slack_webhook
        self.email_config = email_config or {}
        
    async def send_discord_alert(self, alert: AlertMessage) -> bool:
        """Send alert to Discord webhook"""
        if not self.discord_webhook:
            return False
            
        try:
            async with aiohttp.ClientSession() as session:
                embed = {
                    "title": f"🚨 {alert.alert_type.value.upper()} Alert",
                    "description": alert.message,
                    "color": 0xff6b6b,
                    "fields": [
                        {"name": "Symbol", "value": alert.symbol, "inline": True},
                        {"name": "Time", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M"), "inline": True}
                    ]
                }
                
                payload = {"embeds": [embed]}
                
                async with session.post(self.discord_webhook, json=payload) as response:
                    return response.status == 204
                    
        except Exception as e:
            print(f"Discord alert failed: {e}")
            return False
    
    async def send_slack_alert(self, alert: AlertMessage) -> bool:
        """Send alert to Slack webhook"""
        if not self.slack_webhook:
            return False
            
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "text": f"🚨 *{alert.alert_type.value.upper()} Alert*",
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Symbol:* {alert.symbol}\n*Alert:* {alert.message}\n*Time:* {alert.timestamp.strftime('%Y-%m-%d %H:%M')}"
                            }
                        }
                    ]
                }
                
                async with session.post(self.slack_webhook, json=payload) as response:
                    return response.status == 200
                    
        except Exception as e:
            print(f"Slack alert failed: {e}")
            return False
    
    def send_email_alert(self, alert: AlertMessage) -> bool:
        """Send alert via email"""
        if not self.email_config:
            return False
            
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config.get('from_email')
            msg['To'] = self.email_config.get('to_email')
            msg['Subject'] = f"Trading Alert: {alert.alert_type.value}"
            
            body = f"""
            Trading Alert Generated
            
            Symbol: {alert.symbol}
            Alert Type: {alert.alert_type.value}
            Message: {alert.message}
            Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            
            Additional Details:
            {json.dumps(alert.metadata, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config.get('smtp_server', 'smtp.gmail.com'), 
                                self.email_config.get('smtp_port', 587))
            server.starttls()
            server.login(self.email_config.get('from_email'), self.email_config.get('password'))
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            print(f"Email alert failed: {e}")
            return False
    
    async def send_all_alerts(self, alert: AlertMessage) -> Dict[str, bool]:
        """Send alert to all configured channels"""
        results = {
            'discord': False,
            'slack': False,
            'email': False
        }
        
        # Send to Discord
        if self.discord_webhook:
            results['discord'] = await self.send_discord_alert(alert)
        
        # Send to Slack
        if self.slack_webhook:
            results['slack'] = await self.send_slack_alert(alert)
        
        # Send email
        if self.email_config:
            results['email'] = self.send_email_alert(alert)
        
        return results
    
    def create_price_alert(self, symbol: str, price_level: float, alert_type: str = 'above') -> AlertMessage:
        """Create price level alert"""
        message = f"Price alert: {symbol} has reached {price_level}"
        
        return AlertMessage(
            symbol=symbol,
            alert_type=AlertType.PRICE_ALERT,
            message=message,
            timestamp=datetime.now(),
            metadata={'price_level': price_level, 'alert_type': alert_type}
        )
    
    def create_pattern_alert(self, symbol: str, pattern_name: str, confidence: float) -> AlertMessage:
        """Create pattern detection alert"""
        message = f"Pattern detected: {pattern_name} with {confidence:.1f}% confidence"
        
        return AlertMessage(
            symbol=symbol,
            alert_type=AlertType.PATTERN_DETECTED,
            message=message,
            timestamp=datetime.now(),
            metadata={'pattern': pattern_name, 'confidence': confidence}
        )
    
    def create_earnings_alert(self, symbol: str, earnings_date: str, estimate: float) -> AlertMessage:
        """Create earnings alert"""
        message = f"Earnings alert: {symbol} reports on {earnings_date}, estimate: ${estimate}"
        
        return AlertMessage(
            symbol=symbol,
            alert_type=AlertType.EARNINGS_ALERT,
            message=message,
            timestamp=datetime.now(),
            metadata={'earnings_date': earnings_date, 'estimate': estimate}
        )
    
    def create_options_flow_alert(self, symbol: str, unusual_volume: int, put_call_ratio: float) -> AlertMessage:
        """Create options flow alert"""
        message = f"Options flow alert: {symbol} unusual volume {unusual_volume}, P/C ratio: {put_call_ratio:.2f}"
        
        return AlertMessage(
            symbol=symbol,
            alert_type=AlertType.OPTIONS_FLOW,
            message=message,
            timestamp=datetime.now(),
            metadata={'unusual_volume': unusual_volume, 'put_call_ratio': put_call_ratio}
        )

# Configuration templates
CONFIG_TEMPLATES = {
    'discord': {
        'webhook_url': 'YOUR_DISCORD_WEBHOOK_URL'
    },
    'slack': {
        'webhook_url': 'YOUR_SLACK_WEBHOOK_URL'
    },
    'email': {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'from_email': 'your_email@gmail.com',
        'password': 'your_app_password',
        'to_email': 'recipient_email@gmail.com'
    }
}

# Example usage
if __name__ == "__main__":
    # Initialize alert system
    alert_system = MobileAlertSystem(
        discord_webhook="https://discord.com/api/webhooks/YOUR_WEBHOOK_URL",
        slack_webhook="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
        email_config={
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'from_email': 'your_email@gmail.com',
            'password': 'your_app_password',
            'to_email': 'recipient_email@gmail.com'
        }
    )
    
    # Create and send alerts
    alert = alert_system.create_price_alert('AAPL', 150.0, 'above')
    
    # Send to all channels
    import asyncio
    asyncio.run(alert_system.send_all_alerts(alert))
