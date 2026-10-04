"""ROS tests run on a private domain (70 to 79, never 0) and discover peers on this host only."""

import os

domain = os.environ.get("ROS_DOMAIN_ID", "")
if not (domain.isdigit() and 70 <= int(domain) <= 79):
    os.environ["ROS_DOMAIN_ID"] = "73"
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"  # Jazzy and newer
os.environ["ROS_LOCALHOST_ONLY"] = "1"  # Humble
