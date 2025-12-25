# ------------------------------------------------------------
# 🪐 Integrated Vedic Astrology Dashboard (v22)
# ------------------------------------------------------------
# Changes:
# - Integrated BSP (Bhrigu Saral Paddhati) Rules in Tab 4.
# - Displays Active BSP Rules based on Running Age.
# - Includes detailed Zodiac Predictions (Tabs 1 & 5).
# ------------------------------------------------------------

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import BytesIO, StringIO
import json, os
import math
import pytz
from datetime import date, datetime, time

# ==========================================
# 0️⃣ DATA & CONFIGURATION
# ==========================================

# --- Tithi & Dagdha Data ---
TITHI_DATA = {
    1:  {"name": "Pratipada", "dagdha": "Libra & Capricorn", "ruler": "Agni"},
    2:  {"name": "Dwitiya",   "dagdha": "Sagittarius & Pisces", "ruler": "Vidhata or Brahma"},
    3:  {"name": "Tritiya",   "dagdha": "Leo & Capricorn", "ruler": "Gauri"},
    4:  {"name": "Chaturthi", "dagdha": "Taurus & Aquarius", "ruler": "Yama or Ganapati"},
    5:  {"name": "Panchami",  "dagdha": "Gemini & Virgo", "ruler": "Naaga or Serpents"},
    6:  {"name": "Shashthi",  "dagdha": "Aries & Leo", "ruler": "Kartikeya"},
    7:  {"name": "Saptami",   "dagdha": "Cancer & Sagittarius", "ruler": "Surya"},
    8:  {"name": "Ashtami",   "dagdha": "Gemini & Virgo", "ruler": "Rudra"},
    9:  {"name": "Navami",    "dagdha": "Leo & Scorpio", "ruler": "Ambikaa"},
    10: {"name": "Dashami",   "dagdha": "Leo & Scorpio", "ruler": "Dharmaraja"},
    11: {"name": "Ekadashi",  "dagdha": "Sagittarius & Pisces", "ruler": "Rudra"},
    12: {"name": "Dwadashi",  "dagdha": "Libra & Capricorn", "ruler": "Vishnu or Aditya"},
    13: {"name": "Trayodashi","dagdha": "Taurus & Scorpio", "ruler": "Kama Devata"},
    14: {"name": "Chaturdashi","dagdha": "Gemini, Virgo, Sagittarius & Pisces", "ruler": "Kali"},
    15: {"name": "Purnima / Amavasya", "dagdha": "NIL", "ruler": "Pitru Deva"}
}

# --- 1. Transit Data ---
try:
    transit_df = pd.read_csv('transits_full.csv')
    transit_df['timestamp_IST'] = pd.to_datetime(
        transit_df['timestamp_IST'], utc=True, errors='coerce'
    ).dt.tz_convert('Asia/Kolkata')
    transit_df['event_date'] = transit_df['timestamp_IST'].dt.date
    transit_df['year'] = transit_df['timestamp_IST'].dt.year
    unique_years = sorted(transit_df['year'].unique(), reverse=True)
    unique_planets = sorted(transit_df['planet'].unique())
    TRANSIT_DATA_LOADED = True
except Exception as e:
    TRANSIT_DATA_LOADED = False

# --- 2. Karmic Data ---
KARMIC_DATA_CSV_STRING = """Sign,Dharma (right expression),Adharma (distortion),Common Triggers
Aries,"Courage, initiative, protecting the vulnerable, clean competition, swift decision with accountability","Impulsiveness, aggression, picking fights, ego-wins over true wins","Feeling sidelined, slow processes, disrespect"
Taurus,"Steadfastness, stewardship of resources, patience, making beauty tangible, loyalty","Stubbornness, materialism, hoarding, inertia, possessiveness","Sudden change, scarcity fear, threats to comfort"
Gemini,"Curiosity, honest dialogue, learning/teaching, connecting people and ideas, adaptability","Gossip, distraction, superficiality, manipulation via words","Boredom, info overload, fear of “not knowing”"
Cancer,"Care, emotional attunement, homebuilding, protecting lineage, nourishment","Smothering, moodiness, guilt-trips, clannish exclusion","Feeling unsafe, family stress, nostalgia loops"
Leo,"Heart-led leadership, creative radiance, celebrating others, noble pride","Vanity, attention-seeking, drama, authoritarian streak","Being ignored, creative blockage, public slight"
Virgo,"Refinement, service through skill, analysis for healing, integrity in details","Perfectionism, nitpicking, paralysis by analysis, self-criticism","Disorder, unclear standards, time pressure"
Libra,"Fairness, harmonizing, diplomacy, aesthetic balance, principled partnership","People-pleasing, indecision, conflict-avoidance, image over substance","Disharmony, ugliness/chaos, fear of rejection"
Scorpio,"Depth, transformation, truth-telling, loyalty under fire, sacred boundaries","Control, secrecy, vendettas, emotional manipulation","Betrayal, power struggles, taboo shame"
Sagittarius,"Wisdom-seeking, big-picture vision, optimism, ethical adventure, teaching","Dogma, preachiness, risk without responsibility, restlessness","Constraint, hypocrisy, moral injury"
Capricorn,"Duty, long-range building, prudence, tradition with progress, exemplar leadership","Cynicism, workaholism, cold ambition, rigidity","Weak leadership, wasted time, goal drift"
Aquarius,"Humanitarian innovation, systems thinking, principled rebellion, community","Alienation, contrarianism for its own sake, detachment from feelings","Groupthink, injustice, stifling norms"
Pisces,"Compassion, imagination, devotion, unity consciousness, healing presence","Escapism, boundary collapse, martyrdom, deceit (including self-deceit)","Overwhelm, suffering fatigue, unclear roles"
"""
try:
    karmic_df = pd.read_csv(StringIO(KARMIC_DATA_CSV_STRING))
    KARMIC_DATA_LOADED = True
except Exception:
    KARMIC_DATA_LOADED = False

# --- 3. Activation Age Data (Nadi & BSP) ---
ZODIAC_ACTIVATION_CSV = """Ascendant,Zodiac, Activation Year,Age Type
Aries,Leo,7,Nadi
Taurus,Leo,32,Nadi
Gemini,Leo,4,Nadi
Gemini,Leo,5,Nadi
Gemini,Leo,8,Nadi
Gemini,Leo,12,Nadi
Cancer,Leo,25,Nade
Leo,Leo,3,Nadi
Virgo,Leo,36,Nadi
Libra,Leo,25,Nadi
Scorpion,Leo,18,Nadi
Saggitarius,Leo,14,Nadi
Capricorn,Leo,10,Nadi
Aquarius,Leo,25,Nadi
Pisces,Leo,20,Nadi
Aries,Virgo,30,Nadi
Taurus,Virgo,24,Nadi
Gemini,Virgo,25,Nadi
Cancer,Virgo,21,Nadi
Leo,Virgo,28,Nadi
Virgo,Virgo,20,Nadi
Libra,Virgo,38,Nadi
Scorpion,Virgo,17,Nadi
Saggitarius,Virgo,15,Nadi
Capricorn,Virgo,25,Nadi
Aquarius,Virgo,16,Nadi
Pisces,Virgo,13,Nadi
"""
PLANET_ACTIVATION_CSV = """Planet,House, Activation Year,Age Type
SUN,1st House,3,Nadi
SUN,2nd House,5,Nadi
SUN,3rd House,4,Nadi
SUN,3rd House,5,Nadi
SUN,3rd House,8,Nadi
SUN,3rd House,12,Nadi
SUN,4th House,32,Nadi
SUN,5th House,7,Nadi
SUN,6th House,20,Nadi
SUN,7th House,25,Nadi
SUN,8th House,10,Nadi
SUN,9th House,14,Nadi
SUN,10th House,18,Nadi
SUN,11th House,25,Nadi
SUN,12th House,36,Nadi
MERCURY,1st House,17,Nadi
MERCURY,2nd House,15,Nadi
MERCURY,3rd House,25,Nadi
MERCURY,4th House,16,Nadi
MERCURY,5th House,13,Nadi
MERCURY,6th House,30,Nadi
MERCURY,7th House,24,Nadi
MERCURY,8th House,25,Nadi
MERCURY,9th House,21,Nadi
MERCURY,10th House,28,Nadi
MERCURY,11th House,20,Nadi
MERCURY,12th House,38,Nadi
VENUS,1st House,13,Nadi
VENUS,2nd House,32,Nadi
VENUS,3rd House,19,Nadi
VENUS,4th House,30,Nadi
VENUS,5th House,24,Nadi
VENUS,6th House,31,Nadi
VENUS,7th House,12,Nadi
VENUS,8th House,04,Nadi
VENUS,9th House,35,Nadi
VENUS,10th House,39,Nadi
VENUS,11th House,45,Nadi
VENUS,12th House,01,Nadi
MARS,1st House,5,Nadi
MARS,2nd House,13,Nadi
MARS,4th House,18,Nadi
MARS,5th House,06,Nadi
MARS,6th House,27,Nadi
MARS,8th House,24,Nadi
MARS,9th House,17,Nadi
MARS,10th House,18,Nadi
MARS,11th House,23,Nadi
MARS,12th House,42,Nadi
JUPITER,1st House,16,Nadi
JUPITER,2nd House,16,Nadi
JUPITER,2nd House,31,Nadi
JUPITER,3rd House,38,Nadi
JUPITER,4th House,21,Nadi
JUPITER,5th House,18,Nadi
JUPITER,6th House,27,Nadi
JUPITER,7th House,34,Nadi
JUPITER,8th House,41,Nadi
JUPITER,8th House,42,Nadi
JUPITER,9th House,35,Nadi
JUPITER,10th House,32,Nadi
JUPITER,11th House,32,Nadi
JUPITER,11th House,44,Nadi
JUPITER,12th House,45,Nadi
JUPITER,12th House,49,Nadi
SATURN,1st House,04,Nadi
SATURN,1st House,28,Nadi
SATURN,2nd House,39,Nadi
SATURN,2nd House,42,Nadi
SATURN,3rd House,11,Nadi
SATURN,3rd House,56,Nadi
SATURN,4th House,31,Nadi
SATURN,4th House,53,Nadi
SATURN,5th House,19,Nadi
SATURN,5th House,44,Nadi
SATURN,6th House,27,Nadi
SATURN,6th House,65,Nadi
SATURN,7th House,35,Nadi
SATURN,7th House,48,Nadi
SATURN,8th House,32,Nadi
SATURN,8th House,46,Nadi
SATURN,9th House,36,Nadi
SATURN,10th House,25,Nadi
SATURN,10th House,37,Nadi
SATURN,11th House,44,Nadi
SATURN,12th House,03,Nadi
SATURN,12th House,22,Nadi
SATURN,12th House,61,Nadi
"""

BSP_DATA_CSV_STRING = """Planet,BSP Rule #,Timing (Year of Life),Target House (From Planet),Effect / Logic
Saturn,BSP 1,Lifetime,4th,Creates fluctuations (ups & downs) in this house.
Saturn,BSP 3 & 4,Lifetime,1st (Placement),"The Judge. Demands strict discipline here or punishment occurs."
Saturn,BSP 13,20th Year,3rd,Activates 3rd aspect. Focus on effort/siblings.
Saturn,BSP 27,24th Year,4th,Repeats BSP 1 fluctuations specifically at this age.
Saturn,BSP 22,28th Year,6th,"Activates obstacles, service, or health issues."
Saturn,BSP 27,48th Year,4th,Second cycle of BSP 1/27 fluctuations.
Saturn,BSP 26,51st Year,1st (Placement),Major activation of Saturn's placement/karma.
Saturn,BSP 34,Lifetime,10th,Impacts the house of career/karma relative to itself.
Jupiter,BSP 19,22nd Year,9th,"Early activation of luck, father, or higher learning."
Jupiter,BSP 11,32nd Year,5th,"Activates 5th aspect (Progeny, creativity, speculation)."
Jupiter,BSP 32,37th Year,1st / 7th,General activation of Jupiter's axis.
Jupiter,BSP 10,40th Year,9th,"Activates 9th aspect (Dharma, wisdom)."
Jupiter,BSP 36,40th Year,8th,Sudden events or transformation relative to Jupiter.
Jupiter,BSP 18,69th Year,1st (Placement),Late-life activation of Jupiter.
Jupiter,BSP 2,Lifetime,6th & 10th,Creates fluctuations or changes in these houses.
Jupiter,BSP 5,Lifetime,Aries (Sign),"If Jup & Sat are conjunct, the house with Aries requires immense struggle."
Mars,BSP 6,27th Year,1st (Placement),"Mars Wakes Up. High energy, action, or conflict here."
Mars,BSP 29,32nd Year,4th,"Activates 4th aspect (Home, property, mother)."
Mars,BSP 31,36th Year,8th,"Activates 8th aspect (Sudden breaks, surgery, changes)."
Mars,BSP 7,Lifetime,10th,Impacts the karma/action of the 10th house from itself.
Rahu,BSP 21,22nd Year,5th,"Sudden changes in education, romance, or creativity."
Rahu,BSP 30,30th Year,1st (Placement),Major activation of Rahu's obsession/illusion.
Rahu,BSP 8,38th Year,6th,"Impacts enemies, health, or service."
Ketu,BSP 9,24th Year,12th,"Spiritual insights, losses, or exits from that house."
Mercury,BSP 28,24th Year,10th,Career or professional changes linked to intellect.
"""

# ------------------------------------------------------------
# 1️⃣ Astronomical Helpers
# ------------------------------------------------------------

def normalize_degrees(d):
    d = d % 360
    if d < 0: d += 360
    return d

def get_julian_day(date_obj, time_obj, timezone_str):
    local_tz = pytz.timezone(timezone_str)
    dt = datetime.combine(date_obj, time_obj)
    dt_aware = local_tz.localize(dt)
    dt_utc = dt_aware.astimezone(pytz.utc)
    y, m, d = dt_utc.year, dt_utc.month, dt_utc.day
    h = dt_utc.hour + dt_utc.minute/60.0 + dt_utc.second/3600.0
    if m <= 2: y -= 1; m += 12
    a = math.floor(y / 100); b = 2 - a + math.floor(a / 4)
    jd = math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + d + b - 1524.5
    return jd + h / 24.0

def get_positions(jd):
    d = jd - 2451545.0
    w = 282.9404 + 4.70935E-5 * d; e = 0.016709 - 1.151E-9 * d; M = 356.0470 + 0.9856002585 * d
    M = normalize_degrees(M); L = w + M; L = normalize_degrees(L)
    E = M + (180/math.pi) * e * math.sin(math.radians(M)) * (1 + e * math.cos(math.radians(M)))
    x = math.cos(math.radians(E)) - e; y = math.sin(math.radians(E)) * math.sqrt(1 - e*e)
    v = math.degrees(math.atan2(y, x)); lon_sun = normalize_degrees(v + w)
    N = 125.1228 - 0.0529538083 * d; w_moon = 318.0634 + 0.1643573223 * d
    e_moon = 0.054900; M_moon = normalize_degrees(115.3654 + 13.0649929509 * d)
    E0 = M_moon + (180/math.pi) * e_moon * math.sin(math.radians(M_moon)) * (1 + e_moon * math.cos(math.radians(M_moon)))
    E1 = E0 - (E0 - (180/math.pi) * e_moon * math.sin(math.radians(E0)) - M_moon) / (1 - e_moon * math.cos(math.radians(E0)))
    x_moon = math.cos(math.radians(E1)) - e_moon; y_moon = math.sqrt(1 - e_moon*e_moon) * math.sin(math.radians(E1))
    v_moon = math.degrees(math.atan2(y_moon, x_moon)); lon_moon_geo = normalize_degrees(v_moon + w_moon)
    Ms = M; Mm = M_moon; L_moon_mean = N + w_moon + M_moon; D = L_moon_mean - L
    p_lon = -1.274 * math.sin(math.radians(Mm - 2*D)) + 0.658 * math.sin(math.radians(2*D)) - 0.186 * math.sin(math.radians(Ms)) - 0.059 * math.sin(math.radians(2*Mm - 2*D)) - 0.057 * math.sin(math.radians(Mm - 2*D + Ms)) + 0.053 * math.sin(math.radians(Mm + 2*D)) + 0.046 * math.sin(math.radians(2*D - Ms)) + 0.041 * math.sin(math.radians(Mm - Ms)) - 0.035 * math.sin(math.radians(D)) - 0.031 * math.sin(math.radians(Mm + Ms))
    return lon_sun, normalize_degrees(lon_moon_geo + p_lon)

def calculate_dagdha_rashis(b_date, b_time, b_tz):
    try:
        jd = get_julian_day(b_date, b_time, b_tz)
        s_lon, m_lon = get_positions(jd)
        diff = normalize_degrees(m_lon - s_lon)
        tithi_num = int((diff / 12) + 1)
        lookup_index = tithi_num
        if tithi_num > 15:
            lookup_index = tithi_num - 15
            if lookup_index == 0: lookup_index = 15
        dagdha_str = TITHI_DATA[lookup_index]['dagdha']
        if not dagdha_str or dagdha_str == "NIL": return []
        cleaned = dagdha_str.replace(" & ", ",").replace(" and ", ",").replace(", ", ",")
        return [s.strip() for s in cleaned.split(",") if s.strip()]
    except Exception: return []

# ------------------------------------------------------------
# 2️⃣ Standard Helpers
# ------------------------------------------------------------

def get_planet_details(df, planet, target_date):
    if df is None: return None
    p_df = df[df['planet'] == planet].copy()
    past = p_df[p_df['event_date'] <= target_date].sort_values('timestamp_IST', ascending=False)
    future = p_df[p_df['event_date'] > target_date].sort_values('timestamp_IST', ascending=True)
    if past.empty: status_str, prev_str = "Unknown", "No history found"
    else:
        last = past.iloc[0]
        motion = "Retrograde" if last['retrograde'] else "Direct"
        status_str = f"{motion} in {last['zodiac_at_event']}"
        prev_str = f"{last['event']} on {last['event_date']}"
    if future.empty: next_str = "No future events"
    else:
        nxt = future.iloc[0]
        next_str = f"{nxt['event']} on {nxt['event_date']}"
    return {"status": status_str, "prev": prev_str, "next": next_str}

def load_activation_data(csv_content):
    df = pd.read_csv(StringIO(csv_content))
    df[' Activation Year'] = pd.to_numeric(df[' Activation Year'], errors='coerce')
    df.columns = df.columns.str.strip()
    return df

df_zodiac_act = load_activation_data(ZODIAC_ACTIVATION_CSV)
df_planet_act = load_activation_data(PLANET_ACTIVATION_CSV)
df_bsp = pd.read_csv(StringIO(BSP_DATA_CSV_STRING))

def centroid(pts):
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    x = np.append(x, x[0]); y = np.append(y, y[0])
    a = x[:-1] * y[1:] - x[1:] * y[:-1]; A = np.sum(a) / 2
    if abs(A) < 1e-6: return np.mean(x), np.mean(y)
    cx = np.sum((x[:-1] + x[1:]) * a) / (6 * A); cy = np.sum((y[:-1] + y[1:]) * a) / (6 * A)
    return cx, cy

def get_rotated_zodiacs(ascendant):
    idx = SIGNS.index(ascendant)
    return SIGNS[idx:] + SIGNS[:idx]

def calculate_running_age(dob):
    today = date.today()
    completed_years = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return completed_years + 1

def get_current_planet_positions(df, target_date):
    """
    Returns a dictionary mapping Zodiac Sign -> List of (PlanetName, IsRetrograde) tuples
    for the given target_date.
    Example: {'Aries': [('Sun', False), ('Rahu', True)]}
    """
    if df is None: return {}
    positions = {}
    planets = ['Jupiter', 'Saturn', 'Rahu', 'Ketu', 'Mars', 'Venus', 'Sun', 'Mercury', 'Moon']
    
    for p in planets:
        p_df = df[df['planet'] == p].copy()
        # Find the latest event *on or before* target_date
        past = p_df[p_df['event_date'] <= target_date].sort_values('timestamp_IST', ascending=False)
        
        if not past.empty:
            last = past.iloc[0]
            sign = last['zodiac_at_event']
            is_retro = bool(last['retrograde'])
            
            # Append planet to the list for this sign
            if sign not in positions:
                positions[sign] = []
            positions[sign].append((p, is_retro))
            
    return positions

# --- UPDATED BHADAK HELPER (Supports Target Input) ---
def get_bhadak_activation_full(target_sign):
    target_idx = SIGNS.index(target_sign) + 1
    if target_idx in [1, 4, 7, 10]: offset = 11      
    elif target_idx in [2, 5, 8, 11]: offset = 9     
    else: offset = 7                                 
    
    bhadak_idx_raw = (target_idx + offset - 1) % 12
    if bhadak_idx_raw == 0: bhadak_idx_raw = 12
    prim_bhadak_sign = SIGNS[bhadak_idx_raw - 1]
    
    sec_idx_raw = (target_idx + 6 - 1) % 12
    if sec_idx_raw == 0: sec_idx_raw = 12
    sec_bhadak_sign = SIGNS[sec_idx_raw - 1]
    
    magic_num = bhadak_idx_raw 
    h1_idx = (bhadak_idx_raw + magic_num - 1) % 12
    if h1_idx == 0: h1_idx = 12
    magic_1_sign = SIGNS[h1_idx - 1]
    
    return {
        "prim_bhadak": prim_bhadak_sign,
        "sec_bhadak": sec_bhadak_sign,
        "magic_num": magic_num,
        "magic_1": magic_1_sign
    }


# ------------------------------------------------------------
# 3️⃣ Setup & Config
# ------------------------------------------------------------

st.set_page_config(page_title="Vedic Dashboard v12", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] { gap: 24px; }
        .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0 0; padding-top: 10px; padding-bottom: 10px; }
        .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #ff4b4b; }
        div[data-testid="stMetricValue"] { font-size: 1.8rem; }
    </style>
""", unsafe_allow_html=True)

SIGNS = ["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"]
SIGNS_EMOJI = ["♈", "♉", "♊", "♋", "♌", "♍", "♎", "♏", "♐", "♑", "♒", "♓"]

CYCLES = {
    "Cycle 1": list(range(1, 13)), "Cycle 2": list(range(13, 25)),
    "Cycle 3": list(range(25, 37)), "Cycle 4": list(range(37, 49)),
    "Cycle 5": list(range(49, 61)), "Cycle 6": list(range(61, 73))
}

HOUSE_DESCRIPTIONS = {
    1: "Bhadak for Lagna", 2: "Bhadak for Wealth", 3: "Bhadak for Efforts",
    4: "Bhadak for Comforts", 5: "Bhadak for Past-life Merits", 6: "Bhadak for Paying Debts",
    7: "Bhadak for Relationship", 8: "Bhadak for Longevity", 9: "Bhadak for God's Blessing",
    10: "Bhadak for Karma", 11: "Bhadak for Gains", 12: "Bhadak for Spirituality"
}

HOUSE_LABELS = {
    1: "Self", 2: "Family", 3: "Siblings, Servants,\nNeighbour, FIL", 4: "Mother",
    5: "Children,\nP. Grandfather", 6: "Relatives,Mistress,\nPets,Servants",
    7: "Spouse, Buss. Partner,\n2nd child, M. Grandmother", 8: "In-laws",
    9: "Father, Guru, GC,\nSpouse siblings", 10: "MIL, Father(secondary)",
    11: "Eld. Sibling, Friends,\nChildren-in-law,\nUncles/Aunts", 12: "P. Grandmother,\nM. Grandfather"
}

# --- UPDATED ZODIAC TEXT (All 12 Signs) ---
ZODIAC_TEXT = {
    "Aries": """### Key Predictions:
* **Personality:** Wherever Aries sits in the horoscope, the person associated with that house will be **egoistical and Angry** ("I, Me and Myself"). This sign is linked to a **phobia of self-importance**.
* **Parenting:** Aries is connected to the issue of **Foster parenting**.
* **Health:** Aries is linked to **Head injury/accident**. It is also related to **stomach problems** (6-8 relationship from Aries and Scorpio).
* **Relationships:** Aries can have **Tiff relationship** with family members, business partner or spouse. (Taurus and Libra are Maraka).
* **Bhadak (Obstacle/Trap):** For Aries Lagna, the trap (Bhadak) involves running after **gains and wealth** (Aquarius) and seeking **fame and power** (Bhadak for Karma Capricorn is Leo in 5th house of mind).
* **Self-Obsession Trap:** For Aries Lagna, taking a **selfie** or being self-obsessed is a trap (Aquarius is a Bhadak, and Rahu is its co-owner).
* **Spouse:** Spouse of Aries native would **frequently change mind**. Spouse's mind (5th from the 7th house) is Aquarius, which is retrograde in nature.
* **Sun:** Sun is an important planet as it gets Exalted in Aries, occupies Leo in 5th house of mind, and gets Debilitated in 7th house of marriage/partnership. Being egoistical toward spouse or business partner will bring turmoil.
* **Remedy/Preference:** Aries Ascendant natives would love **massage of lower back or feet** with Oil/Ghee (Saturn/Jupiter).
* **Secrets:** Wherever Aries sits, there is a chance of an **extra-marital affair** for that house relative.
* **Influence:** Wherever Aries sits, that Relative from Capricorn would be influential in relative life (Bhadak for Scorpio is Capricorn).""",
    
    "Taurus": """### Key Characteristics:
* **Nature:** Wherever Taurus sits, native is **impatient and lazy** (if afflicted). It is 4th from 11th (storage of gains).
* **Appreciation:** Wherever Taurus or Venus sits, the person would either **deserve or crave for Appreciation**. Morality becomes a concern.
* **Stability:** Taurus is a Fixed Rashi, signifying Stability, but wherever it sits, that house will have **stability issues**.
* **Scandal:** Wherever Taurus sits, the **relative gets cheated** or the native is the one cheating/being cheated. If afflicted, it brings scandal (especially with Rahu transit).
* **Blindness:** Person might be **blinded** for that house signification (e.g., in 9th house, blinded by religious ceremonies).
* **Health:** Afflicted Taurus can give **dental, face or chin issues, premature ejaculation**.
* **Areas of Test:** Finances/Wealth, family issues, jewelry and administration.
* **Ego:** Wherever Taurus sits, after the materialization of that house signification, it would give **huge ego**.
* **Objects:** **Jewelry/Necklace** technique: Relative might have a break in necklace or lose it.
* **Debt:** Any planet in Taurus shows a **debt** regarding the planet's Karaktatwas and medical expenses. Past life debt of jewelry to spouse if connected to 6, 8, 12, 7.
* **Money:** Using free-will in money matter can be detrimental (Mercury is Maraka). Excessive money thirst impacts married life (6-8 relationship between Taurus/Libra).
* **Travel:** Clothes might get torn or spotted during short travel (2-12 relationship).
* **Preferences:** Connected to orchids, lilies, and roses. Spouse happy if gifted earrings or necklace.

### Bhadak & Maraka:
* **Hard Work:** Need for Hard Work (Capricorn is Bhadak) is must. Saturn is Yogkaraka.
* **Obstacles:** Difficulty in **religious ceremonies** or delays/non-completion where Taurus sits.

### Relationships:
* **Spouse:** Karmic connection if connected to 3, 6, 8, 12 and 7. Spouse can be vengeful or secretive. She would **always be on phone**. Husband/spouse would be **pulling the blanket while sleeping** (4-6 AM).
* **Pets:** Virgo in 5th house indicates they would be **crazy to have pets**.
* **Family:** Someone in family has tendency to **cheat, deceit and double speak** on money matter (Rahu exalted in Gemini).

### Remedies:
* **Pray to Nandi** by telling your problem in his ears for 21 days.
* Use **Musk attar** on the body part signified by the house where Taurus sits.

### Activation Age:
* **28th or 34th year.**
* **36 years** (Fixed sign activation).
""",

    "Gemini": """### Key Characteristics:
* **Nature:** Wherever Gemini sits, it gives **hypocrisy and Dual nature** for that house signification and gives the relative an **inner desire for pomp and show (Marketing)**. If connected to a karmic house, they would forever be looking for a soulmate.
* **Remedy:** Gemini Lagna should **treat their daughter as a queen** to get Laxmi's blessing, as it is a Vishnu Lagna.
* **Identity:** If afflicted, the person might have **dual names**. If Mercury is afflicted, they would have multiple names.
* **Destiny:** Natives would have at least **two or three ups/downs** in their life (Aquarius in 9th house). Saturn is co-lord of 9th house, making a tough destiny. Timing would be off for religious ceremonies.
* **Money:** Moon is Markesh; if afflicted, **money will flow like water**.
* **Health:** Trouble digesting food. If Mars/Ketu is afflicted, can suffer from **IBS/colitis** (Scorpio in 6th). Gemini also represents Lungs. Hands would be quick, but later in age, it would become a problem.
* **Reproductive/Hidden:** Issues with **Pelvic Area** or hidden relationships. May not be sexually satisfied or have issues like **premature ejaculation** (Taurus in 12th). Knee issues (Capricorn in 8th).
* **Psychology:** Wherever Gemini sits, there is an issue of the **subconscious mind**, leading to psychological problems.
* **Karma:** Gemini is a dual Rashi; karma is a mix of **past life (0-15°)** and **current life (15-30°)**. Must be careful with free will; if afflicted, they create new karma.
* **Spirituality:** Ego about **charity or donation** (Taurus in 12th). Chance of fraud when donating. Astrologers with this ascendant should **do sadhana** before predicting. Karma relates to knowledge, donation, moksha, and **letting go** (Pisces) of controlling money and family.
* **Supernatural:** House may have **Vaastu dosh** (Virgo in 4th). If Mercury is afflicted, may feel spirit visits, hear walking/anklet sounds, or feel someone sitting on the bed.
* **Objects:** The **Necklace** is very important. Native might start giving pearl necklaces as gifts (Cancer in 2nd, Bhadak Taurus in 12th).

### Relationships:
* **Spouse:** **Religion/Guru** might be the focal point of heated discussion. Problems with priest/ceremonies during marriage. Spouse would have **dreams of dead people** (Scorpio in 12th). Dress color of spouse causes heated conversation.
* **Children:** **Imbalanced relation**. Native should avoid ego. Communication issues if Leo is afflicted.
* **Siblings:** Extreme loyalty, but chances of cheating with elder sibling. Sibling might change name if Mercury is in 3rd house. If Moon is afflicted, elder sibling faces downfall (Taurus is Bhadak).
* **Guru:** Will meet a guru hungry for **money and fame**.
* **Father:** Issues with inheritance of property (Virgo in 4th).
* **Mother-in-Law:** Marriage may be spoiled by mother/mother-in-law. She may have a **grabbing nature**, shed crocodile tears, hide things, and keep calling like a broken record/phobia.
* **In-Laws:** Sudden attitude change (Scorpio). Spouse's sibling might be crooked.
* **Relatives:** Secretive, vindictive, potential prostate issues, associated with **black magic/nazar dosh** (Scorpio in 6th).

### Bhadak & Obstacles:
* **Jupiter:** Critical planet (Markesh and Badhkesh). **Preaching ethics** or moral values to spouse/business partner creates problems. Area of religion/guru (Sagittarius) is area of conflict.
* **Debt Obstacle:** Bhadak for Scorpio (6th) is Cancer (2nd). Hence **Mother, family, and property** will prevent the native from paying debts.
* **Karma Obstacle:** Bhadak for Pisces (10th) is Virgo (4th). Hence **Property, vehicles, relatives, mother, and health** prevent the native from doing prescribed karma.

### Important Ages:
* **3.5 - 5.5 years:** Early childhood (Relatives, Gemini, Mercury, Aries, Scorpio activate).
* **11.5 - 12.5 years:** Pre-teen (Gemini, Mercury, Jupiter activate).
* **17.5 - 18.5 years:** Teenage (Major events).
* **33.5 years:** Gemini, its lord, and associated yoga get activated.
""",

    "Cancer": """### Key Characteristics:
* **Symbolism:** Cancer is associated with **security, Fortune, throne, chair** and signifies **whatever you have in the heart**.
* **Ambition:** Cancer ascendants have ambition of being king but yearn to be at home or in homeland. They also have a darker side which would be unknown to people.
* **Health:** If afflicted, there could be an **eating disorder**. This is confirmed if the Moon is also afflicted or debilitated (Moon is exalted in the 2nd house of food but debilitated in Scorpio). Wherever Cancer sits, there is a possibility that the **end of life** for the native or relative signified by Cancer could be painful or sudden.
* **Food Preference:** A Cancer Ascendant person would likely **eat rice**.
* **Psychology:** Addiction could be a problem if Mars is afflicted (Scorpio in the 5th house of mind). The combination of **Gemini (12th house, subconscious)** and **Scorpio (5th house, mind)** creates a 6/8 relationship, leading to **psychological issues and phobias**.
* **Secrets & Fame:** They have **psychological issues with Name and Fame** and are too sensitive about it. May have hidden Perverted sex fantasy (Aquarius in 8th). If Saturn is aspecting the Lagna, Moon Lagna, 7th lord, or Venus, there is either no sex, perversion in sex, or irregular sex. The native must be careful regarding **sexual pleasure, charity, and donation** as there are chances of it coming in public. They should **not curse**, as that might have a chance of coming true (Leo, which is Aatma Karaka, is in the 2nd house).
* **Obsessions:** The native is **obsessed with necklace/jewelry** as Gemini (pomp and show) comes in the 12th house of past life. Native might have secretly been to tantrik for some sadhna related to name, fame and money (Aquarius in Scorpio).
* **Karma & Obstacles:** The native's karma is to **Avoid being self-centered** ("I, Me and Myself") and **control aggression**. The native wants to **excel in their work** (Aries in 10th house). Bhadak for doing karma is Aquarius in 8th house. Hence desire of inheritance and gains by shortcut, native's social circle would be impediment in his karma.
* **Debts:** The native needs to pay debts to **guru, relatives, religion, and father**. The **obstacles (Bhadak)** preventing debt payment are **charity, bed pleasures, siblings, hypocrisy, pomp and show** (due to Gemini in the 12th house).
* **Spiritual Trap:** Biggest mistake for them when they are pursuing spiritual path they go after name and fame. This is because 5th house of mantra is Scorpio but 8th house have Aquarius (desires and fame) and bhadak house have Taurus (wealth).
* **Philosophy:** Their 5th house of intellect has Scorpio (9th from 9th house) and 9th house have Pisces (philosophy and past life house), they have a philosophy/thesis regarding everything spiritual.
* **Remedy:** The native should be careful regarding **scandal** as false accusations can arise (Aquarius in the 8th house, co-lorded by Rahu). **Massage the glutes** or **give leather purses or shorts** to express love and get kudos from Cancer Lagna native.
* **Warning:** For Cancer ascendant, when the necklace of children's spouse is broken, then there would be trouble in married life (Taurus necklace technique).

### Exaltation/Debilitation:
* **Property:** When Cancer or the Moon sits in a chart, that relative has a **hunger, greed, or propensity toward the property** or would face issues related to property, because Mars (the karaka of property) gets debilitated in Cancer.
* **Anger:** Cancer natives have possible **Anger issues** (Mars is debilitated in cancer and Scorpio is in 5th house of mind).
* **Mother-in-law:** Going to be an issue or influential in marriage because **Aries falls in the 10th house** (Mars is exalted in natural 10th house).

### Bhadaka & Maraka:
* **Bhadaka:** **Taurus** is in bhadak house, and **Venus is the Badhkesh**. This placement means the native struggles with **bad advice** (e.g., women wearing white pearls can give bad advice).
* **Maraka:** Laidback attitude toward food and money can cause problem. Cheating or being unethical in actions would spoil marriage and business.

### Relationships:
* **Spouse:** **Capricorn in the 7th house** means that spouse might have tendency of grabbing (if afflicted Capricorn/Saturn afflicted).
* **Children:** Trouble in married life would start when the **necklace of the children's spouse is broken** (Taurus is the bhadak). The relative would be responsible for the state of their marriage.
* **In-Laws:** Would be **twisted or involved in sadhana** (Aquarius in the 8th house). The spouse's siblings can have a **manglik dosha**.
* **Family:** **Leo falls in the 2nd house**, so the person would be **dominating in the family**.
* **Siblings:** **Virgo** in the 3rd house means the native has **pending karma toward their siblings**. **Gemini in the 12th house** also signifies past life karma toward siblings.
* **Guru:** Native's guru will be an **intense past life** figure they are indebted to (Sagittarius in 6th house in Pisces in 12th house).
* **Others:** The native has **huge pending karma regarding relatives and subordinates**. They should be careful with their direct/fiery communication style (Virgo in 3rd house). Relatives or uncles can be racist or believe in casteism.

### Drishti Dosh (Affliction by Aspect):
* If **Saturn is in the 11th house (Taurus)** or conjoined with the Moon(LL), it causes **Drishti badha** (due to Saturn being the natural bhadak in the bhadak house).
* **Venus + Moon(LL) combination** also causes Drishti badha.
* If the lord of the 7th house (Saturn) aspects the **bhadak sthan** (11th house), it is a Drishti Dosh.
* For a **Chara Rashi** like Cancer, if the Lagna has a malefic and Mars is in the 7th house, **Drishti dosh** is caused by the Devatas (Mars aspects the 2nd house of Leo).

### Activation Ages:
* **14-15 years:** The person would either **travel or change in life**.
* **25 years:** Cancer activates (Nadi).
* **27 years:** **Infatuation** may occur.
* **33 years:** The **energy from the past life** would come into the native's life.
* **38 years:** There is a chance of **frequent hospitalization**.
""",

    "Leo": """### Key Characteristics:
* **Fixed Fire:** Leo is a Fixed, Fiery sign ruled by the Sun, emphasizing **authority, royalty**.
* **Royalty:** Leo signifies Authority and those associated with it love to behave or be treated like a **King**.
* **Criticism:** 2nd house has Virgo, hence Leo ascendant might **fight when eating**, speak ill or get critical about food, relatives, co-workers.
* **Ego:** Wherever Leo sits, the person associated with that house can have **huge ego**. Leo is linked to the trap of seeking richness, fame, and power.
* **Timing:** Wherever Leo or the Sun sits, that relative would either be **laid back (like a king)** or the results would be **seen later in life**.
* **Conceiving:** Wherever Leo or the Sun sits, that relative would have an **issue in conceiving** or the birth would be delayed.
* **Health:** Wherever Leo sits, that relative can have issues with **eyes, lungs, or heart**. Leo is associated with the bed (Purva Phalguni/Uttara Phalguni).
* **Hair:** Wherever Leo sits, the person would **lose hair and go bald early** in life.
* **Karma:** 10th House is Taurus. Karma is to **fulfill family life** and not be involved in the trap of money and jewelry. Mastering ego, family life, administration, and fame.
* **Wealth:** 2nd House is Virgo. Leo person gets **selfish with the acquisition of wealth**. If Virgo is afflicted by 7th lord (Saturn/Rahu) and Mercury is retrograde, spouse would cheat the native of documents. If Venus is debilitated in 2nd, chance of women being ill-treated.
* **Sex Life:** Wherever Leo sits, that relative would have an **issue with their sex life**.
* **Career:** If Leo is afflicted, there would be a **downfall in career** or position of power (6/8 relationship).
* **Inheritance:** Native might lose or denounce inheritance because of escapist nature or idealistic views.
* **Gold:** Wherever Leo or Sun sits, relative would have an issue/story with **gold**, especially during marriage.
* **Death:** Leo people have a **dramatic death** because Pisces is in the 8th house.
* **Charity:** Leo Ascendant people are **generous** in their donation, charity, or gift.
* **Traps:** Trapped by richness, fame, and power. Tendency to **judge the mother-in-law**, twisting communication, and gossiping are traps.
* **Transformation:** If Sun/Leo connects with 3, 8, 12 and Rahu/Ketu, there is a connection with **forest and caves**. Visiting a forest/cave activates Aquarius (Rahu/Saturn) for a huge bounce back.
* **Spouse Gift:** Spouse of Leo/Aquarius native needs **something royal** as a gift. Also, buy shoes/anklets and put a black thread on the ankle.
* **Separation:** If Leo is in 7th house, married life won't be smooth (Sun is separative).
* **Education:** If Sun/Leo is afflicted, studies **won't get completed**.
* **End of Life:** Might be **bed-ridden** for sometime, especially at end of life.
* **Government:** If Leo is 11th from Arudh lagna, gain from government.
* **Secrets:** 8th from Leo is escapist/silent relative. Inheritance, dark arts, tantra are hidden for the king.

### Bhadaka and Maraka:
* **Bhadaka:** **Aries** (9th house) is Bhadak Rashi, **Mars** is Badhkesh. Obstruction is native's **huge ego** ("I, Me and Myself").
* **Maraka:** **Gemini and Aquarius** are Maraka rashis (Mercury and Saturn).
* **Drishti Dosh:** If Sun with Saturn, or Saturn/Mars aspect Lagna.

### Relationships:
* **Spouse:** **Aquarius in 7th house**. Native would never know nature of married life. Married life starts late or spouse settled late. Spouse might have habit of **changing clothes** (Taurus in 4th from 7th) or **sleeping in different bed** (Pisces in 8th). Spouse might hurt feet with glass/stone in water.
* **Mother:** **Scorpio in 4th house**. Difficulties maintaining rulership. Intense insecurity. If Mars/Ketu afflicted, mother may have groin issues.
* **Siblings:** **Libra in 3rd house**. Like to hold hands/shoulder while walking with siblings. Should not bring ego while talking to siblings.
* **Family:** Property might have issue of **snakes**. In-laws can have escapist attitude.
""",

    "Virgo": """### Key Characteristics:
* **Dual Earth:** Virgo is ruled by Mercury, focusing on service, analytical thinking, and perpetual search. Passion for literature, reading books, poems. Barren sign, sometimes issues conceiving.
* **Conflict:** Wherever Virgo sits, the signification of that house or relative would be the **reason for the fight**. Malefic in 6th intensifies fighting nature.
* **Service:** Native has taken birth for **service, repay past life debt**, and mastering shad-ripus.
* **Marriage:** Wherever Virgo sits, marriage of that person is challenged or they might have a **never married woman or widow** in their life.
* **Cheating:** Any planet in Virgo, that signification the person would **cheat or get cheated**. Once caught, will never accept and blame others (6th from Virgo is Aquarius). If retrograde Mercury in 6th, native would cheat wife and hide mistress.
* **Temple:** Home temple can be **untidy** or have mice droppings if aspected by Saturn/Rahu.
* **Decision Making:** Karmic decision making. Free-will or vengeance backfires.
* **Career:** **Multiple careers** or side jobs (Gemini in 10th).
* **Blood/Property:** Might face **blood-related problems** or **property-related issues**.
* **Inheritance:** Land/inheritance gets **split between siblings** (Scorpio in 3rd). If not split equal, karma formed.
* **Bed:** Bed should have problem (Leo in 12, 6, or 8) -> afflicted Purva/Uttara Phalguni.
* **Travel/Neighbors:** Finicky about travel, vehicles, neighbors (Scorpio in 3rd, Moon debilitated). Extensive planning or no planning.
* **Pets:** If they lose pets, it gives hard times/trouble (Aquarius in 6th).
* **Health:** **Gastric issues** (Leo in 12th). **Lower leg issues** (Aquarius in 6th).
* **Love:** Will meet someone they fall in love with but **won't be able to unite** (Leo in 12th).
* **Donation:** Need to donate food, clothes, ornament.
* **Karma:** Improve **communication**, avoid hypocrisy and loose talk (Gemini in 10th).
* **Spirituality:** Need to build or do something with money for **religion** (Taurus in 9th).
* **Appreciation:** Appreciates a **heel massage**.

### Bhadak and Marak:
* **Health:** Natural Baadha house is 6th. If 6th afflicted, **health issues 4x normal**. Rahu/Ketu in 6/8/12 or 3/7 makes health major issue.
* **Jupiter:** Badhkesh and Markesh. Pisces is Bhadak. Retrograde Jupiter causes unimaginable things related to property/partnership.
* **Mars:** Absolute **Maraka** (rules 3rd/8th). Wherever it sits, creates huge problem.
* **Malefic Order:** MARS > Ketu > Rahu/Saturn.

### Marital Issues:
* **Infidelity:** Chance of **extra-marital affair**. Jupiter (Badhkesh) rules 4th (Character) and 12th (Spouse). 6th from 7th is Aquarius (sin). Jupiter with Saturn/Rahu gives marital issues.
* **Conjugal Life:** Venus debilitated in Virgo negatively impacts conjugal life. Long dry spells.
* **Mercury+Venus:** Conjunction in Virgo won't give happy married life.
* **Beds:** Spouse and native might end up in **different beds** or uneven mattresses (Pisces Bhadak).
* **Secrets:** Native might **hide a mistress** if Retro Mercury in 6th.
* **Spouse:** Spouse coming from **immediate past life**. 8th house Aries -> spicy speech. Spouse secretive, escapist attitude.

### Relationships:
* **Sibling:** Vindictive, issues with travel/property documents.
* **Mother:** Spiritual, native might have issue with rituals. Mother's blessing important.
* **Children:** Knees/neck issues. Hypocrisy/Free will impacts children's health.
* **In-Laws:** Might be **selfish** (Aries in 8th).

### Activation Ages:
* **25 years:** Important for Virgo/Gemini.
* **32-33 years:** Significant changes.

### Remedies:
* **Oil Ritual:** **Til or mustard oil** essential.
* **Marital:** Two cotton wicks intertwined, light ghee diya.
* **Clutter:** Clutter clearing on Saturday (if Jup/Sat associated). Keep home pristine.
* **Donation:** Donate food, clothes, ornaments. Learn to give (Taurus in 9th). Feed milk/Thandai or jalebi to aghora (if Rahu in 9th).
""",

    "Libra": """### Key Characteristics:
* **Balance:** Desire to **restore balance**. Soul born in Aries, completed in Libra. Wherever Libra sits, native needs to achieve balance; imbalance will happen until mastered.
* **Principles vs Money:** In business, either Taurus (money) or Libra (principle) is spoiled (6-8 relationship).
* **Character Test:** Wherever Libra sits, there is a **test of character** (4th from 4th).
* **Lack of Principles:** Wherever Libra sits, that person/relative **won't have any principles**. Accused of harboring black money.
* **Imbalance:** Desire runs in circles (6th from Libra is Pisces).
* **Health:** Cut in body part signified by Libra house; medical expenses related to it.
* **Family Tragedy:** Family might have history of someone **dying within 7 years** of birth. Chronic disease, sudden death, phobias in family.
* **Sex:** Sex is very important; must come with feeling of satisfying the other person.
* **Financial:** Accused of harboring **black money**.
* **Food:** Extremities in food choices. If Mars/Ketu in 2nd (Scorpio), finicky about food/salt.
* **Dark Arts:** Someone in family working with **dark arts/Sadhana**.
* **Past Life Love:** Wherever Libra sits (e.g., 5th), love affair repeating for several lifetimes.
* **Karma:** Extremely loyal to karma (Cancer in 10th).
* **Shoes/Fortune:** If 6th afflicted, reversal of fortunes by **changes in shoes** (Pisces) or obsession with shoes/feet problems.
* **Scandal:** If Rahu+Jupiter in 12th, married life of relative finished. Sun+Rahu gives **scandal**.
* **Appreciation:** Appreciate **massage of head or lower back**.
* **Planets:** Sun debilitated, Saturn exalted.
* **Ego/Children:** Ego and children (Leo is Bhadak) are issues.

### Relationships:
* **Guru:** Involved in **pomp and show**.
* **Spouse:** **Aries in 7th**. Good home life (Capricorn in 4th) leads to good married life.
* **Elder Sibling:** Test for character or infidelity.
* **Children:** Difficulty with **second child**; chance of abortion or difficulty conceiving.
* **Family:** Spicy speech, psychological phobias, dark arts. Complain about inheritance.

### Activation Age:
* **33.5 years.**
""",

    "Scorpio": """### Key Characteristics:
* **Inheritance:** Scorpio people think that they **deserve inheritance**. Wherever Scorpio sits, scandals or accusations related to that house.
* **Religion:** Customize religion as per convenience (Cancer in 9th Bhadak).
* **Secrets:** Property and mother are issues kept **hidden from everyone** (Aquarius in 4th). Mother will suck energy on property issue.
* **Curse:** If afflicted/Mars afflicted, it is a curse. Mars+Saturn may relate to murder in past life.
* **Difficulty:** Relative signified by Scorpio is **complicated or difficult** (Ketu). Sudden change in attitude.
* **Career:** Leo in 10th - like to talk about it. If Saturn afflicted, often **late for work**.
* **Food:** Never like food in large quantity (Moon debilitated). Tend to spill food. Finicky about salt.
* **Psychology:** Embedded psychological issues (Gemini in 8th).
* **Frustration:** Multiple things happen at once, frustrating the native (Ketu).
* **Memory:** Never forget relatives/significations where Mars/Ketu sits.
* **Health:** Breathing/asthma in mid-life. Women chance of breast cancer. Foul language/Family scandal if afflicted.
* **Karma:** Justifying actions, power, ego. Spend lavishly on children's education.
* **Remedy:** Give spouse flowers. Massage ears. Sitting in cave activates Aquarius (transformation).
* **Evil Eye:** Through food or dress (Virgo/Scorpio in 2nd).

### Bhadak and Maraka:
* **Bhadak:** **Cancer** (9th house).
* **Maraka:** **Sagittarius and Taurus**. Venus is important (past life/Markesh).
* **Advice:** Venus+Mercury in Gemini gives bad advice.

### Relationships:
* **Marriage:** Issue because **Libra in 12th**. Love story from past life but not culminating in marriage (Pisces in 5th). "Hide n seek" in marriage.
* **Spouse:** **Taurus in 7th**. Spouse argumentative (Gemini in 2nd from 7th). Massage lower back. Spouse shoulder problem.
* **In-Laws:** Communication issues. Sister-in-law plays role (Gemini in 8th).
* **Mother:** Special relationship (Moon debilitated). Mother **crooked nature** (Aquarius in 4th). Massage knees of mother.
* **Elder Sibling:** Spouse of sibling might sleep in separate room (Pisces). Sibling divorce/separation. Fight at 40-45.
* **Children:** Past life related. Conceived very early/late/before marriage. Triangular relation.
* **Father:** Unequal distribution of property.
""",

    "Sagittarius": """### Key Characteristics:
* **Opposition:** Opposes Gemini (Mercury/Jupiter enemies).
* **Traps:** Religion, higher education, racism.
* **Nature:** Fighting/stubborn nature. If afflicted, racist/superiority complex about knowledge.
* **Communication:** Issue for karma if Mercury afflicted (6/8 with Virgo).
* **Karmic Trap:** Communication, wife, career change, food, family, ego clashes, image consciousness.
* **Debt:** Grabbing mentality, family, food prevent debt payment (Taurus in 6th).
* **Karma Prevention:** Property, sex, mother, charity, spirituality prevent karma (Pisces Bhadak).
* **Finance:** Careful with money karma. Family member grabbed religious land/ignored kuldevta (Capricorn in 2nd, Jupiter debilitated).
* **Career:** Can change career path (Virgo in 10th). Avoid gossip/jealousy (affects children). Mercury in 11th = secretive about income.
* **Fame:** Hidden ego, want authority/fame. Fortune supports later (Leo in 9th). Deepest desire to be Guru.
* **Occult:** Meet tantrik, gain siddhis (Scorpio in 12th).
* **Health:** Blood in urine (Taurus in 6th). Careful about **sugar consumption**. Phobia of heights (Scorpio in 12th). Vitamin B-12 check.
* **Vaastu:** North-west problematic. Green color predominant. Remedy: Bury 1g gold in center.
* **Travel:** Absurdity in short travel (Aquarius in 3rd). Stop at doorstep or U-turn.
* **Adharma:** Rahu in Sagittarius or Rahu+Jupiter = prone to Adharma.
* **Jewelry:** Necklace/beads loan/debt. Break in relative's necklace = marriage difficulty.
* **Body:** Shoulder area problem/fitting (Scorpio in 12th). Spouse should massage shoulders.

### Bhadaka and Maraka:
* **Bhadak:** **Gemini** (7th house). Master communication. Worst sin: cheating partner/spouse.
* **Maraka:** **Capricorn** (2nd house). Tendency to move away from Dharma.

### Relationships:
* **Spouse:** **Gemini in 7th**. Child goes to bed with parents. Wife complains about books/phone. Avoid making spouse a friend. Spouse from past life.
* **Siblings:** **Aquarius in 3rd**. Sibling loves travel abroad, late when traveling. Say one thing do another.
* **Father/Guru:** Jupiter in 6th = sin regarding Guru. Father important.
* **Mother:** **Pisces in 4th**. Keep changing Vaastu. Property/spiritual issues.
* **In-Laws:** Mercury rules 7th/10th -> Spouse/SIL/Husband nature same. Spouse sibling settles late.
* **Children:** **Aries in 5th**. Past life connection. Speech issues or break in education.

### Activation Age:
* **35–37 years** (esp 35.5).
""",

    "Capricorn": """### Key Characteristics:
* **Nature:** Grabbing nature.
* **Communication:** Relative asks same question twice/thrice (Pisces in 3rd).
* **Karma:** Wherever Capricorn sits, 20% flavor of 10th house (Karma). Burning desire for business (Libra in 10th). Karma 12th from 11th -> spend karma to get gains.
* **Authority:** Issue with Authority/Government.
* **Money:** Shrewd way of earning (Aquarius in 2nd).
* **Spending:** Spend on remedies (Taurus in 5th).
* **Rituals:** Must perform **annual shraadh** (Virgo in 9th). Visit Samshan important.
* **Vaastu:** Very important (Aries in 4th, Mars Badhkesh).
* **Physical:** Sweat more. Hearing problem (Gemini in 6th).
* **Ethics:** Never ill-treat/steal from Brahmins.
* **Property:** Focal point in relationship. Lot of discussion (Mars debilitated in Cancer).
* **Trap:** Imbalance in karma, grabbing, seeking power.
* **Ego:** Psychological problem is Ego (Leo in 8th). Do not abuse power. Sun makes or breaks.
* **Planet:** Jupiter most important (debilitated in Capricorn).

### Bhadaka and Maraka:
* **Bhadak:** **Scorpio** (11th house). Sexual compatibility issue. Pitru Dosh. Phobias.

### Relationships:
* **Marriage:** Sexual compatibility issue (Scorpio Bhadak). Ego prevents intimacy. Good partnership if Venus not afflicted.
* **Mother-in-Law:** Interferes in marriage.
* **Mother:** Dominate household (Aries in 4th).
* **Siblings:** Fight or mistake out of free will (Gemini in 6th). Sibling loves travel abroad. Instrumental in inheritance. Sibling marriage break due to intimacy/family.
* **Children:** From past life. If Venus/Sun afflicted, getting son is issue.
* **Family:** Knees problem.

### Activation Age:
* **28th year** (Unforgettable).
* **31st year** (Name/Fame if Jupiter in 3/7).
""",

    "Aquarius": """### Key Characteristics:
* **Nature:** Fixed, ruled by Saturn/Rahu. Desires, selfishness, hidden aspects, inconsistency.
* **Selfishness:** Selfish regarding house where Aquarius sits.
* **Secrecy:** Hide things, shrewdness, cunningness. Hide at least one aspect.
* **Complication:** Relative complicated/difficult (Scorpio in 10th).
* **Timing:** Timing always off.
* **Repetition:** Ask same question twice/thrice (Pisces in 2nd).
* **Gait:** Peculiar way of walking.
* **Ethics:** If not just, finished. Unjust to Libra (Bhadak) activates obstruction. Libra in 9th -> need to be ethical.
* **Physical:** Nerve issues. Victory/Achievements.
* **Finance:** Bankruptcy/downfall couple times (Scorpio in 10th).
* **Karma:** From past life (Scorpio in 10th).
* **Challenge:** Lagna considered bad (11th Bhadak, Virgo 8th, Scorpio 10th).
* **Jewelry:** Might wear **fake jewelry** (Gemini in 5th).
* **Traps:** Hiding self, desires. Moh maya for fame (Leo in 7th).
* **Remedy:** Buy **shoes/anklets with black thread** for spouse. Take care of feet (Saturn in 2nd). Sesame oil.

### Bhadaka and Maraka:
* **Bhadak:** **Libra** (9th house). Venus is Badhkesh. Keep Venus in check.
* **Maraka:** **Pisces and Leo**. Idealism/Ego cause conflict.

### Relationships:
* **Spouse:** **Leo in 7th**. Settle/start late. Needs royal gift. Habit of changing clothes.
* **Love:** **Gemini in 5th**. Incomplete love life. Phobias.
* **Mother:** **Taurus in 4th**. Emotional about money/family. Trapped by mother issues/property cheating.
* **Children:** Problem with knees (Capricorn in 12th) if Saturn afflicted.

### Activation Age:
* **33.5 years.**
""",

    "Pisces": """### Key Characteristics:
* **Nature:** Confusion and Desire. Churning desire.
* **Change:** Frequent changes in profession, religion/faith. Reversal and repetitive patterns.
* **Self:** Always reinventing themselves.
* **Advice:** Give up desire to have everything.
* **Sleep:** Want **pillow in lower body** (Aquarius in 12th).
* **Ego:** Think they are perfect. Afflicted -> mental problems, emotional weakness, ego.
* **Food:** Love **fish** or dosha (if 2nd lord in Pisces).
* **Fear:** Fear of death/imprisonment where Aquarius sits.

### Karmic Debt and Afflictions:
* **Pending Karma:** Heavy pending karma where Pisces sits.
* **Past Life:** 6th house Pisces -> determined by past life. 12th lord in Pisces -> tired, running after moksha due to loss in past life.
* **Fear:** Fear of death, Imprisonment.

### Relationships:
* **Marriage:** **Virgo in 7th**. Perpetual search/dissatisfaction. Fights, divorce, two marriages. Second woman picture. Spouse changes religion/guru/profession.
"""
}

# Chart Geometry
A=(0,0); B=(12,0); C=(12,12); D=(0,12) 
E=(6,0); F=(12,6); G=(6,12); H=(0,6) 
P1=(9,9); P2=(3,9); P3=(3,3); P4=(9,3) 
C_C = (6,6) 

HOUSES = {
    1:  [G, P1, C_C, P2], 4:  [H, P2, C_C, P3],
    7:  [E, P3, C_C, P4], 10: [F, P4, C_C, P1],
    2:  [D, G, P2],       3:  [D, H, P2],
    5:  [H, A, P3],       6:  [A, E, P3],
    8:  [E, B, P4],       9:  [B, F, P4],
    11: [F, C, P1],       12: [C, G, P1]
}

# ------------------------------------------------------------
# 4️⃣ Sidebar Controls
# ------------------------------------------------------------

with st.sidebar:
    st.title("🔮 Profile Setup")
    st.info("Enter birth details to configure the entire dashboard.")
    
    # 1. Name Input
    name = st.text_input("Name of Person", value="Native")

    # 2. Lagna Selector
    lagna = st.selectbox("Ascendant (Lagna)", SIGNS, index=0)
    
    # 3. Dagdha & Logic Details Toggle
    show_dagdha = st.radio("Show Dagdha Rashi?", ["Yes", "No"], index=0)

    # Initialize variables to avoid NameError if hidden
    dob = None
    tob = None
    tz_sel = None
    running_age = None

    if show_dagdha == "Yes":
        # 3. DOB & TOB Selector
        dob = st.date_input("Date of Birth", value=date(1988, 11, 19), min_value=date(1945, 1, 1), max_value=date(2035, 1, 1))
        tob = st.time_input("Time of Birth", value=time(12, 00))
        
        # 4. Timezone
        common_timezones = ['Asia/Kolkata', 'UTC', 'America/New_York', 'Europe/London']
        tz_sel = st.selectbox("Timezone", common_timezones + list(set(pytz.all_timezones) - set(common_timezones)), index=0)

        # 5. Age Metric
        running_age = calculate_running_age(dob)
        st.metric("Running Age", f"{running_age} Years")
    else:
        st.info("Time details hidden. Dagdha Rashi and Age-based activations will be disabled.")
    


    st.divider()
    st.caption("Developed for Mridul Gupta • v12")

# ------------------------------------------------------------
# 5️⃣ Main GUI
# ------------------------------------------------------------

st.title(f"🪐 Blank Chart Analysis Tool: {name} ({lagna} Lagna)")
st.caption("Based on Sunil John's Course on Blank chart prediction")

# Calculate Dagdha Rashis dynamically
if dob and tob and tz_sel:
    user_dagdha_rashis = calculate_dagdha_rashis(dob, tob, tz_sel)
else:
    user_dagdha_rashis = []

tab1, tab3, tab4, tab5, tab6 = st.tabs([
    "✨ Core Analysis", 
    "🪐 Transits", 
    "⚡ Activation Ages",
    "♈ Zodiac Reference",
    "📝 My Notes"
])

# ==========================================
# TAB 1: Chart (UPDATED LOGIC)
# ==========================================
with tab1:
    # --- 0. INITIALIZE STATE & DATA ---
    house_options = list(HOUSE_DESCRIPTIONS.values())
    
    # Ensure session state is initialized for the selector before we use it
    if 'life_area_selector' not in st.session_state:
        st.session_state.life_area_selector = house_options[0]
        
    current_selector_val = st.session_state.life_area_selector
    selected_house_label = current_selector_val # Fix for NameError in Tab 6
    
    # --- 1. CALCULATE ANALYSIS (Before Rendering) ---
    selected_house_num = house_options.index(current_selector_val) + 1
    is_default_view = (selected_house_num == 1)
    
    lagna_idx = SIGNS.index(lagna)
    target_idx = (lagna_idx + selected_house_num - 1) % 12
    target_sign = SIGNS[target_idx]
    
    analysis = get_bhadak_activation_full(target_sign)

    # Determine Chart Houses for highlighting/text
    sign_to_chart_house = {}
    for h in range(1, 13):
        s_idx = (lagna_idx + h - 1) % 12
        sign_to_chart_house[SIGNS[s_idx]] = h
        
    house_prim = sign_to_chart_house[analysis['prim_bhadak']]
    house_sec = sign_to_chart_house[analysis['sec_bhadak']]
    house_m1 = sign_to_chart_house[analysis['magic_1']]

    # --- 2. LAYOUT COLUMNS (3-COLUMN LAYOUT) ---
    c_left, c_mid, c_right = st.columns([1, 1, 1], gap="medium")

    # --- 3. LEFT COLUMN: CHART & CONTROLS ---
    with c_left:
        # A. CONTROLS (Moved to Top)
        c_sel, c_res = st.columns([3, 1])
        with c_sel:
            st.selectbox(
                "Select Life Area:", 
                house_options,
                key="life_area_selector",
                label_visibility="collapsed"
            )
        with c_res:
            def reset_selection():
                st.session_state.life_area_selector = house_options[0]
            st.button("Reset", on_click=reset_selection, help="Reset to Bhadak for Lagna")

        # B. PLOT CHART
        plt.close("all")
        fig, ax = plt.subplots(figsize=(7,7))
        ax.set_xlim(0,12); ax.set_ylim(0,12); ax.axis("off")
        ax.plot([0,12,12,0,0],[0,0,12,12,0],'k-',lw=2)
        ax.plot([0,12],[0,12],'k-',lw=1); ax.plot([0,12],[12,0],'k-',lw=1)
        ax.plot([0,6],[6,12],'k-',lw=1); ax.plot([6,12],[12,6],'k-',lw=1)
        ax.plot([12,6],[6,0],'k-',lw=1); ax.plot([6,0],[0,6],'k-',lw=1)

        # Highlight Logic
        if house_prim in HOUSES:
            pts = HOUSES[house_prim]; pts_closed = pts + [pts[0]]; bx, by = zip(*pts_closed)
            ax.fill(bx, by, color="#FF9999", alpha=0.6, zorder=1) 
        if is_default_view and house_sec in HOUSES:
            pts = HOUSES[house_sec]; pts_closed = pts + [pts[0]]; bx, by = zip(*pts_closed)
            ax.fill(bx, by, color="#FFCCF2", alpha=0.6, zorder=1)
        if is_default_view and house_m1 in HOUSES:
            pts = HOUSES[house_m1]; pts_closed = pts + [pts[0]]; bx, by = zip(*pts_closed)
            ax.fill(bx, by, color="#FFD700", alpha=0.5, zorder=1) 

        current_transits = get_current_planet_positions(transit_df, date.today())

        sign_idx_for_house = [(lagna_idx + (h - 1)) % 12 for h in range(1, 13)]
        for h, pts in HOUSES.items():
            pts_closed = pts + [pts[0]]; cx, cy = centroid(pts_closed)
            xl, yl = (0, 0.45); xh, yh = (0, -0.1); xz, yz = (0, -0.5)
            s_idx = sign_idx_for_house[h-1]; sign_num = s_idx + 1; sign_name = SIGNS[s_idx]
            label_text = HOUSE_LABELS.get(h, "")
            
            # Dagdha Check
            is_dagdha = sign_name in user_dagdha_rashis
            sign_color = "gray"; sign_weight = "normal"
            
            if is_dagdha:
                sign_color = "#D32F2F"; sign_weight = "bold"
                # Star (*), Shift Right of Number
                ax.text(cx + 0.25, cy - 0.1, "*", ha="center", va="center", fontsize=20, color='red', fontweight='bold', zorder=5)
            
            ax.text(cx+xl, cy+yl, label_text, ha="center", va="center", fontsize=7, color="red", alpha=0.6, zorder=2)
            ax.text(cx+xh, cy+yh, str(sign_num), ha="center", va="center", fontsize=14, fontweight="bold", color="navy", zorder=3)
            ax.text(cx+xz, cy+yz, sign_name, ha="center", va="center", fontsize=9, 
                    color=sign_color, fontweight=sign_weight, alpha=0.8, zorder=3)
            
            if sign_name in current_transits:
                p_list = current_transits[sign_name]
                p_strs = [f"{p[:2]}{'(R)' if r else ''}" for p, r in p_list]
                full_p_str = ", ".join(p_strs)
                ax.text(cx, cy - 0.9, full_p_str, ha="center", va="center", fontsize=7.5, color="#006400", fontweight="bold", zorder=4)

        st.pyplot(fig)
        
        # C. LEGEND (Bottom of Left Column)
        st.caption("Note: Planets shown in the chart are Transit Planets.")
        
        # Legend (No Title)
        st.markdown(f"🔴 **Primary:** {analysis['prim_bhadak']} (House {house_prim})")
        if is_default_view:
            st.markdown(f"🌸 **Natural:** {analysis['sec_bhadak']} (House {house_sec})")
            st.markdown(f"🟡 **BSP Rule:** {analysis['magic_1']} (House {house_m1})")

        if user_dagdha_rashis:
            st.warning(f"🔥 **Dagdha Rashis:** {', '.join(user_dagdha_rashis)}")
        else:
            st.info("No Dagdha Rashis active.")

        with st.expander("ℹ️ Key Terms"):
            st.markdown("""
            **Dagdha Rashi:** Signs that have low energy due to your birth Tithi.
            
            **Badhak:** The planet or house causing hidden obstacles in your life.
            """)




    # --- 4. MIDDLE COLUMN: INFO & PREDICTIONS ---
    with c_mid:
        # Predictions at top
        with st.expander(f"📖 Read Predictions for {lagna}", expanded=True):
             st.markdown(ZODIAC_TEXT.get(lagna, "Content coming soon..."))

# ==========================================
# TAB 2: Karmic Patterns
# ==========================================
    # --- 5. RIGHT COLUMN: ACTIVATION & KARMA ---
    with c_right:
        # --- CURRENT ACTIVATION SECTION (Moved from Middle) ---
        st.markdown("##### Current Activation")
        if running_age:
            ages_to_show = [running_age, running_age - 1]
            age_labels = ["Current Year", "Previous Year"]
            
            for age, label in zip(ages_to_show, age_labels):
                if age < 1: continue
                with st.expander(f"{label} (Age {age})", expanded=(label=="Current Year")):
                    found_any = False
                    
                    # 0. BCP House Activation
                    # Calculate active details for this age
                    bcp_house = ((age - 1) % 12) + 1
                    rotated_signs_bcp = get_rotated_zodiacs(lagna)
                    bcp_zodiac = rotated_signs_bcp[bcp_house - 1]
                    st.info(f"🏠 **BCP Activation:** House {bcp_house} ({bcp_zodiac})")
                    found_any = True

                    # 1. Zodiac Activation
                    if 'Activation Year' in df_zodiac_act.columns:
                        z_hits = df_zodiac_act[(df_zodiac_act['Ascendant'] == lagna) & (df_zodiac_act['Activation Year'] == age)]
                        for _, row in z_hits.iterrows():
                            st.info(f"✨ **Zodiac ({row['Age Type']}):** {row['Zodiac']}")
                            found_any = True
                    # 2. Planet Activation
                    if 'Activation Year' in df_planet_act.columns:
                        p_hits = df_planet_act[df_planet_act['Activation Year'] == age]
                        for _, row in p_hits.iterrows():
                            st.info(f"🪐 **Planet ({row['Age Type']}):** {row['Planet']} in {row['House']}")
                            found_any = True
                    # 3. BSP Rules
                    if 'Timing (Year of Life)' in df_bsp.columns:
                        bsp_hits = df_bsp[df_bsp['Timing (Year of Life)'].astype(str).str.match(rf"^{age}(\D|$)", na=False)]
                        for _, row in bsp_hits.iterrows():
                           st.info(f"📜 **BSP Rule:** {row['Planet']} - {row['BSP Rule #']} (Target: {row['Target House (From Planet)']})\n\n{row['Effect / Logic']}")
                           found_any = True
                    if not found_any: st.caption("No specific activations found.")
        else:
            st.warning("Enter DOB to see activations.")

        st.markdown("---")
        
        # Karmic Patterns
        if KARMIC_DATA_LOADED:
            with st.expander("⚖️ Karmic Debts & Duties", expanded=False):
                target_houses_map = {6: HOUSE_LABELS[6], 8: HOUSE_LABELS[8], 12: HOUSE_LABELS[12], 3: HOUSE_LABELS[3]}
                target_signs = []
                for h in target_houses_map.keys():
                    t_idx = (lagna_idx + h - 1) % 12; target_signs.append(SIGNS[t_idx])
                
                filtered_df = karmic_df[karmic_df['Sign'].isin(target_signs)].set_index('Sign')
                ordered_df = filtered_df.loc[target_signs].reset_index()
                ordered_df['House'] = list(target_houses_map.keys())
                
                final_cols = ['House', 'Sign', 'Adharma (distortion)', 'Common Triggers', 'Dharma (right expression)']
                st.dataframe(ordered_df[final_cols], use_container_width=True, hide_index=True)
        else:
            st.warning("Karmic data unavailable.")

# ==========================================
# TAB 3: Transits
# ==========================================
with tab3:
    st.subheader("Planetary Transit Viewer")
    if TRANSIT_DATA_LOADED:
        c_filters, c_table = st.columns([1, 3])
        with c_filters:
            st.markdown("### Filters")
            try: default_year_index = unique_years.index(2025)
            except ValueError: default_year_index = 0 
            selected_year = st.selectbox("Select Year:", unique_years, index=default_year_index)
            selected_planets = st.multiselect("Select Planets:", unique_planets, default=["Jupiter", "Saturn"])
        with c_table:
            if not selected_planets: st.warning("Select at least one planet.")
            else:
                filtered_transits = transit_df[(transit_df['year'] == selected_year) & (transit_df['planet'].isin(selected_planets))].copy().sort_values(by='timestamp_IST')
                if not filtered_transits.empty: st.dataframe(filtered_transits[['timestamp_IST', 'planet', 'event', 'zodiac_at_event', 'retrograde']], use_container_width=True, hide_index=True)
                else: st.info(f"No events in {selected_year}.")
    else: st.error("Transit data file not found.")

# ==========================================
# TAB 4: Activation Ages (Combines BCP + Nadi + BSP)
# ==========================================
with tab4:
    if running_age is None:
        st.warning("Please enable 'Show Dagdha Rashi?' and enter Date of Birth to view Activation Ages.")
    else:
        # --- NEW LAYOUT: 3 COLUMNS ---
        c_bcp, c_nadi, c_bsp = st.columns(3)

        # --- COLUMN 1: BCP HOUSE ACTIVATION ---
        with c_bcp:
            st.subheader("BCP House Activation")
            st.caption(f"Current Running Age: {running_age}")
            
            rotated_signs = get_rotated_zodiacs(lagna)
            df = pd.DataFrame({
                "House": list(range(1, 13)), "Zodiac": rotated_signs,
                "Cycle 1": CYCLES["Cycle 1"], "Cycle 2": CYCLES["Cycle 2"],
                "Cycle 3": CYCLES["Cycle 3"], "Cycle 4": CYCLES["Cycle 4"],
                "Cycle 5": CYCLES["Cycle 5"], "Cycle 6": CYCLES["Cycle 6"]
            })
            cycle_ranges = [(1,12), (13,24), (25,36), (37,48), (49,60), (61,72)]
            active_house = ((running_age - 1) % 12) + 1
            active_cycle = None; found_cycle = False
            for i, (start, end) in enumerate(cycle_ranges):
                if start <= running_age <= end:
                    active_cycle = list(CYCLES.keys())[i]; found_cycle = True; break
            if not found_cycle: active_cycle = "Cycle 6"
            active_zodiac = rotated_signs[active_house - 1]

            cols_to_show = ["House", "Zodiac"]
            if active_cycle: cols_to_show.append(active_cycle)
            
            # Filter to ONLY the active house row
            df_active_bcp = df[df['House'] == active_house].copy()
            
            st.dataframe(
                df_active_bcp[cols_to_show], 
                use_container_width=True, 
                hide_index=True
            )

        # --- COLUMN 2: NADI ACTIVATION ---
        with c_nadi:
            st.subheader("♈ Nadi Activation")
            st.caption(f"Calculated based on Birth Year: {dob.year}")
            
            # 1. Zodiac Activation
            st.markdown("#### Zodiac Activation")
            df_z_user = df_zodiac_act[df_zodiac_act['Ascendant'] == lagna].copy()
            if df_z_user.empty: 
                st.warning(f"No data for {lagna}.")
            else:
                df_z_user['Activation Calendar Year'] = (df_z_user['Activation Year'] + dob.year).astype(int)
                st.dataframe(
                    df_z_user[['Ascendant', 'Zodiac', 'Activation Year', 'Activation Calendar Year']], 
                    use_container_width=True, 
                    hide_index=True
                )
                
            st.markdown("---")
            
            # 2. Planetary Activation
            st.markdown("#### Planetary Activation")
            df_p_all = df_planet_act.copy()
            df_p_all['Activation Calendar Year'] = (df_p_all['Activation Year'] + dob.year).astype(int)
            
            current_year = date.today().year
            min_y = int(dob.year)
            max_y = int(dob.year + 100)
            default_start = max(min_y, current_year - 5)
            default_end = min(max_y, current_year + 5)
            
            sel_range = st.slider(
                "Filter Year Range", 
                min_value=min_y, 
                max_value=max_y, 
                value=(default_start, default_end)
            )
            mask = (df_p_all['Activation Calendar Year'] >= sel_range[0]) & (df_p_all['Activation Calendar Year'] <= sel_range[1])
            df_p_filt = df_p_all[mask].copy().sort_values(by='Activation Calendar Year')
            
            if df_p_filt.empty:
                st.info(f"No activations in this range.")
            else:
                st.dataframe(
                    df_p_filt[['Planet', 'House', 'Activation Year', 'Activation Calendar Year']], 
                    use_container_width=True, 
                    hide_index=True
                )

        # --- COLUMN 3: BSP RULES ---
        with c_bsp:
            st.subheader("📜 BSP Rules Link")
            st.caption("Planetary activation based on age.")
            
            # Filter for Current Age
            active_bsp = df_bsp[df_bsp['Timing (Year of Life)'].str.contains(str(running_age), na=False)]
            lifetime_bsp = df_bsp[df_bsp['Timing (Year of Life)'].str.contains("Lifetime", na=False)]
            
            st.markdown(f"#### 🔥 Active Rules (Age {running_age})")
            if not active_bsp.empty:
                for index, row in active_bsp.iterrows():
                    st.info(f"**{row['Planet']} (Rule {row['BSP Rule #']}):** {row['Effect / Logic']} (Target: {row['Target House (From Planet)']})")
            else:
                st.write("No specific BSP rule active for this year.")
            
            st.markdown("---")
            
            st.markdown("#### ♾️ Lifetime Rules")
            with st.expander("Show Lifetime Rules", expanded=False):
                st.dataframe(lifetime_bsp[['Planet', 'Target House (From Planet)', 'Effect / Logic']], use_container_width=True, hide_index=True)

            with st.expander("View All BSP Rules", expanded=False):
                st.dataframe(df_bsp, use_container_width=True)

# ==========================================
# TAB 5: Reference
# ==========================================
with tab5:
    st.subheader("Zodiac Characteristics")
    if 'sign_to_show' not in st.session_state: st.session_state.sign_to_show = None
    c1, c2, c3 = st.columns(3)
    cols = [c1, c2, c3]
    for i, sign in enumerate(SIGNS):
        with cols[i % 3]:
            if st.button(f"{SIGNS_EMOJI[i]} {sign}", use_container_width=True): 
                st.session_state.sign_to_show = sign 
    if st.session_state.sign_to_show:
        sign = st.session_state.sign_to_show
        @st.dialog(f"About {sign}")
        def show_sign_info():
            st.title(f"{SIGNS_EMOJI[SIGNS.index(sign)]} {sign}")
            st.markdown(ZODIAC_TEXT.get(sign, "No text available.")) 
            if st.button("Close"): st.session_state.sign_to_show = None; st.rerun()
        show_sign_info()

# ==========================================
# TAB 6: My Notes
# ==========================================
with tab6:
    st.subheader("📝 Personal Astrology Notes")
    st.caption("Values from your selections are pre-filled below. Add your analysis and download it.")
    
    active_house_num = ((running_age - 1) % 12) + 1 if running_age else "N/A"
    rotated_signs = get_rotated_zodiacs(lagna)
    active_zodiac_name = rotated_signs[((running_age - 1) % 12)] if running_age else "N/A"

    dob_str = str(dob) if dob else "Not Provided"
    tob_str = str(tob) if tob else "Not Provided"
    dagdha_str = ', '.join(user_dagdha_rashis) if user_dagdha_rashis else 'None'
    age_str = str(running_age) if running_age else "N/A"

    # --- Build Activation Summary Text ---
    activation_text = ""
    if running_age:
        act_lines = []
        # 1. Zodiac
        if 'Activation Year' in df_zodiac_act.columns:
            z_hits = df_zodiac_act[(df_zodiac_act['Ascendant'] == lagna) & (df_zodiac_act['Activation Year'] == running_age)]
            for _, row in z_hits.iterrows(): act_lines.append(f"- Zodiac: {row['Zodiac']} ({row['Age Type']})")
        # 2. Planet
        if 'Activation Year' in df_planet_act.columns:
            p_hits = df_planet_act[df_planet_act['Activation Year'] == running_age]
            for _, row in p_hits.iterrows(): act_lines.append(f"- Planet: {row['Planet']} in {row['House']} ({row['Age Type']})")
        # 3. BSP
        if 'Timing (Year of Life)' in df_bsp.columns:
            bsp_hits = df_bsp[df_bsp['Timing (Year of Life)'].astype(str).str.match(rf"^{running_age}(\D|$)", na=False)]
            for _, row in bsp_hits.iterrows(): act_lines.append(f"- BSP Rule: {row['Planet']} (Rule {row['BSP Rule #']})")
        
        if act_lines: activation_text = "\n".join(act_lines)
        else: activation_text = "No specific activations for this age."
    else:
        activation_text = "Age not available."

    system_summary = f"""--- ASTROLOGY DASHBOARD REPORT ---
Generated on: {date.today()}

== PROFILE DETAILS ==
Name: {name}
Ascendant (Lagna): {lagna}
Date of Birth: {dob_str}
Time of Birth: {tob_str}
Dagdha Rashis: {dagdha_str}
Current Running Age: {age_str}

== BCP STATUS ==
Active House: {active_house_num}
Active Zodiac: {active_zodiac_name}

== CURRENT ACTIVATIONS (Age {age_str}) ==
{activation_text}

== BHADAK ACTIVATION ==
Aspect: {selected_house_label}
Target: {target_sign}
Primary Bhadak: {analysis['prim_bhadak']}
Secondary Bhadak: {analysis['sec_bhadak']}
Activation 1: {analysis['magic_1']}
"""
    
    c_note_1, c_note_2 = st.columns([1, 2])
    
    with c_note_1:
        st.markdown("#### System Profile")
        st.text_area("Auto-Generated Data (Read Only)", value=system_summary, height=400, disabled=True)
    
    with c_note_2:
        st.markdown("#### Your Observations")
        user_notes = st.text_area("Enter your analysis here...", height=400, placeholder="Type your notes about this chart, transits, or karmic patterns here...")
    
    full_report = system_summary + "\n== USER NOTES ==\n" + user_notes
    
    st.download_button(
        label="📥 Download Full Report (TXT)",
        data=full_report,
        file_name=f"Astrology_Notes_{name.replace(' ', '_')}_{date.today()}.txt",
        mime="text/plain"
    )