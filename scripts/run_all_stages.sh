#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
#  EurekaAI — Full Curriculum Pipeline Runner
#  Runs Stage 0 through 6 sequentially with logging.
#
#  Usage:
#    chmod +x scripts/run_all_stages.sh
#    ./scripts/run_all_stages.sh
#    ./scripts/run_all_stages.sh --start 2   # Resume from stage 2
# ─────────────────────────────────────────────────────────────────────

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

START_STAGE=${1:-0}
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="$LOG_DIR/run_all_${TIMESTAMP}.log"

echo -e "${CYAN}"
cat << 'EOF'
  ______                _         _    ___ 
 |  ____|              | |       / \  |_ _|
 | |__   _   _ _ __ ___| | __  / _ \  | | 
 |  __| | | | | '__/ _ \ |/ / / ___ \ | | 
 | |____| |_| | | |  __/   < / /   \ \| | 
 |______|\__,_|_|  \___|_|\_/_/     \_\___|
EOF
echo -e "${NC}"
echo -e "${BLUE}🧠 EurekaAI — Full Curriculum Run${NC}"
echo -e "   Start stage: ${YELLOW}${START_STAGE}${NC}"
echo -e "   Log: ${MASTER_LOG}"
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python not found. Please activate your virtualenv.${NC}"
    exit 1
fi

# Check Ollama (optional)
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Ollama detected${NC}"
else
    echo -e "${YELLOW}⚠️  Ollama not detected — synthetic data will be skipped${NC}"
fi

echo ""

STAGE_NAMES=(
    "0: 🍼 신생아 (Newborn)"
    "1: 🧸 유아기 (Toddler)"
    "2: 📚 초등학교 (Elementary)"
    "3: 🔢 중학교 (Middle)"
    "4: 📐 고등학교 (High)"
    "5: 🎓 대학교 (University)"
    "6: 🌐 사회인 (Social)"
)

SUCCESS_COUNT=0
FAIL_COUNT=0

for i in "${!STAGE_NAMES[@]}"; do
    STAGE=$i
    if [ "$STAGE" -lt "$START_STAGE" ]; then
        echo -e "  ${YELLOW}⏭  Skipping Stage ${STAGE_NAMES[$i]}${NC}"
        continue
    fi

    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Stage ${STAGE_NAMES[$i]}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    STAGE_LOG="$LOG_DIR/stage${STAGE}_${TIMESTAMP}.log"

    if python run.py --stage "$STAGE" 2>&1 | tee "$STAGE_LOG" | tee -a "$MASTER_LOG"; then
        echo -e "\n${GREEN}✅ Stage ${STAGE} complete${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo -e "\n${RED}❌ Stage ${STAGE} failed — check ${STAGE_LOG}${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo -e "${YELLOW}Continuing to next stage...${NC}"
    fi
done

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Curriculum Complete!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  ${GREEN}✅ Succeeded: ${SUCCESS_COUNT}${NC}"
echo -e "  ${RED}❌ Failed:    ${FAIL_COUNT}${NC}"
echo ""

# Print final status
python run.py --status
