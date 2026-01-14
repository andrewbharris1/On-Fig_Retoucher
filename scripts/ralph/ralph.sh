#!/bin/bash
#
# Ralph for Claude Code
# An autonomous AI agent loop that runs Claude Code repeatedly until all PRD items are complete.
# Based on Geoffrey Huntley's Ralph pattern: https://ghuntley.com/ralph/
#
# Usage: ./ralph.sh [max_iterations]
#

set -e

# Configuration
MAX_ITERATIONS=${1:-10}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PRD_FILE="$PROJECT_ROOT/prd.json"
PROGRESS_FILE="$PROJECT_ROOT/progress.txt"
PROMPT_FILE="$SCRIPT_DIR/prompt.md"
ARCHIVE_DIR="$PROJECT_ROOT/archive"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    if ! command -v claude &> /dev/null; then
        log_error "Claude CLI not found. Please install it first."
        exit 1
    fi

    if ! command -v jq &> /dev/null; then
        log_error "jq not found. Install with: brew install jq (macOS) or apt install jq (Linux)"
        exit 1
    fi

    if [ ! -f "$PRD_FILE" ]; then
        log_error "prd.json not found at $PRD_FILE"
        log_info "Create a PRD first, or copy prd.json.example to prd.json"
        exit 1
    fi

    if [ ! -f "$PROMPT_FILE" ]; then
        log_error "prompt.md not found at $PROMPT_FILE"
        exit 1
    fi
}

# Archive previous run if branch changed
archive_previous_run() {
    local current_branch=$(jq -r '.branchName // empty' "$PRD_FILE")

    if [ -f "$PROGRESS_FILE" ] && [ -n "$current_branch" ]; then
        # Check if there's existing progress from a different feature
        local last_branch=$(head -1 "$PROGRESS_FILE" 2>/dev/null | grep -oP '(?<=Branch: ).*' || echo "")

        if [ -n "$last_branch" ] && [ "$last_branch" != "$current_branch" ]; then
            local archive_name=$(date +%Y-%m-%d)-${last_branch//\//-}
            mkdir -p "$ARCHIVE_DIR/$archive_name"

            [ -f "$PROGRESS_FILE" ] && cp "$PROGRESS_FILE" "$ARCHIVE_DIR/$archive_name/"

            log_info "Archived previous run to $ARCHIVE_DIR/$archive_name"

            # Reset progress file for new feature
            echo "Branch: $current_branch" > "$PROGRESS_FILE"
            echo "Started: $(date)" >> "$PROGRESS_FILE"
            echo "---" >> "$PROGRESS_FILE"
        fi
    fi
}

# Setup feature branch
setup_branch() {
    local branch_name=$(jq -r '.branchName // empty' "$PRD_FILE")

    if [ -z "$branch_name" ]; then
        log_warn "No branchName in prd.json, using current branch"
        return
    fi

    local current_branch=$(git branch --show-current)

    if [ "$current_branch" != "$branch_name" ]; then
        if git show-ref --verify --quiet "refs/heads/$branch_name"; then
            log_info "Switching to existing branch: $branch_name"
            git checkout "$branch_name"
        else
            log_info "Creating new branch: $branch_name"
            git checkout -b "$branch_name"
        fi
    fi
}

# Get next incomplete story
get_next_story() {
    jq -r '
        .userStories
        | map(select(.passes != true))
        | sort_by(.priority // 999)
        | first
        | if . then .id else empty end
    ' "$PRD_FILE"
}

# Get story details
get_story_details() {
    local story_id=$1
    jq -r --arg id "$story_id" '
        .userStories[] | select(.id == $id) |
        "Story: \(.id)\nTitle: \(.title)\nDescription: \(.description // "N/A")\nAcceptance Criteria:\n\(.acceptanceCriteria // [] | map("- " + .) | join("\n"))"
    ' "$PRD_FILE"
}

# Check if all stories are complete
all_stories_complete() {
    local incomplete=$(jq '[.userStories[] | select(.passes != true)] | length' "$PRD_FILE")
    [ "$incomplete" -eq 0 ]
}

# Count stories
count_stories() {
    local total=$(jq '.userStories | length' "$PRD_FILE")
    local complete=$(jq '[.userStories[] | select(.passes == true)] | length' "$PRD_FILE")
    echo "$complete/$total"
}

# Build the prompt for Claude Code
build_prompt() {
    local story_id=$1
    local story_details=$(get_story_details "$story_id")
    local progress=""

    if [ -f "$PROGRESS_FILE" ]; then
        progress=$(cat "$PROGRESS_FILE")
    fi

    local base_prompt=$(cat "$PROMPT_FILE")

    cat <<EOF
$base_prompt

---

## Current Story

$story_details

---

## Progress from Previous Iterations

$progress

---

## Instructions

1. Implement ONLY the story above (ID: $story_id)
2. Run quality checks (typecheck, tests) before marking complete
3. If checks pass, update prd.json to set passes: true for story $story_id
4. Append any learnings to progress.txt
5. Commit your changes with a descriptive message
6. Output <promise>COMPLETE</promise> when the story is fully implemented and verified
EOF
}

# Run a single iteration
run_iteration() {
    local iteration=$1
    local story_id=$2
    local story_count=$(count_stories)

    log_info "=========================================="
    log_info "Iteration $iteration/$MAX_ITERATIONS"
    log_info "Stories complete: $story_count"
    log_info "Working on: $story_id"
    log_info "=========================================="

    local prompt=$(build_prompt "$story_id")

    # Run Claude Code with the prompt
    # --print: non-interactive mode, prints output
    # --dangerously-skip-permissions: auto-approve tool use (use with caution)
    # You may want to remove --dangerously-skip-permissions for safer operation

    cd "$PROJECT_ROOT"

    echo "$prompt" | claude --print --dangerously-skip-permissions 2>&1 | tee "/tmp/ralph-iteration-$iteration.log"

    local exit_code=${PIPESTATUS[1]}

    if [ $exit_code -ne 0 ]; then
        log_error "Claude Code exited with error code $exit_code"
        return 1
    fi

    # Check if story was marked complete
    local story_passes=$(jq -r --arg id "$story_id" '.userStories[] | select(.id == $id) | .passes' "$PRD_FILE")

    if [ "$story_passes" = "true" ]; then
        log_success "Story $story_id completed!"
        return 0
    else
        log_warn "Story $story_id not yet complete, will retry in next iteration"
        return 0
    fi
}

# Main loop
main() {
    log_info "Starting Ralph for Claude Code"
    log_info "Max iterations: $MAX_ITERATIONS"
    log_info "Project root: $PROJECT_ROOT"

    check_prerequisites
    archive_previous_run
    setup_branch

    # Initialize progress file if it doesn't exist
    if [ ! -f "$PROGRESS_FILE" ]; then
        local branch_name=$(jq -r '.branchName // "unknown"' "$PRD_FILE")
        echo "Branch: $branch_name" > "$PROGRESS_FILE"
        echo "Started: $(date)" >> "$PROGRESS_FILE"
        echo "---" >> "$PROGRESS_FILE"
    fi

    local iteration=1

    while [ $iteration -le $MAX_ITERATIONS ]; do
        # Check if all stories are complete
        if all_stories_complete; then
            log_success "=========================================="
            log_success "All stories complete!"
            log_success "=========================================="
            echo "<promise>COMPLETE</promise>"
            exit 0
        fi

        # Get next story to work on
        local story_id=$(get_next_story)

        if [ -z "$story_id" ]; then
            log_success "No more stories to process!"
            echo "<promise>COMPLETE</promise>"
            exit 0
        fi

        # Run iteration
        run_iteration $iteration "$story_id"

        iteration=$((iteration + 1))

        # Small delay between iterations
        sleep 2
    done

    log_warn "Reached max iterations ($MAX_ITERATIONS)"
    log_info "Stories complete: $(count_stories)"

    local incomplete=$(jq -r '[.userStories[] | select(.passes != true) | .id] | join(", ")' "$PRD_FILE")
    if [ -n "$incomplete" ]; then
        log_warn "Incomplete stories: $incomplete"
    fi
}

main "$@"
