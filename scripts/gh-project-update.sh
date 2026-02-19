#!/bin/bash
# Update cascade-rs GitHub Project item status
# Usage: ./gh-project-update.sh <feature_id> <status>
# Status: backlog | in-progress | verify | done

set -e

FEATURE_ID="$1"
STATUS="$2"
OWNER="DonSqualo"
PROJECT_NUM="${CASCADE_PROJECT_NUM:-1}"  # Set via env or default

if [ -z "$FEATURE_ID" ] || [ -z "$STATUS" ]; then
    echo "Usage: $0 <feature_id> <status>"
    echo "  feature_id: e.g., primitive::box"
    echo "  status: backlog | in-progress | verify | done"
    exit 1
fi

# Get project field info
get_field_info() {
    gh project field-list "$PROJECT_NUM" --owner "$OWNER" --format json
}

# Get item ID by title
get_item_id() {
    gh project item-list "$PROJECT_NUM" --owner "$OWNER" --format json | \
        jq -r ".items[] | select(.content.title == \"$FEATURE_ID\") | .id"
}

# Map status to option ID
get_status_option() {
    local status_name
    case "$STATUS" in
        backlog)     status_name="Todo" ;;
        in-progress) status_name="In Progress" ;;
        verify)      status_name="In Review" ;;  # Or custom
        done)        status_name="Done" ;;
        *)           echo "Unknown status: $STATUS"; exit 1 ;;
    esac
    
    gh project field-list "$PROJECT_NUM" --owner "$OWNER" --format json | \
        jq -r ".fields[] | select(.name==\"Status\") | .options[] | select(.name==\"$status_name\") | .id"
}

echo "Updating $FEATURE_ID to $STATUS..."

ITEM_ID=$(get_item_id)
if [ -z "$ITEM_ID" ]; then
    echo "Error: Item '$FEATURE_ID' not found in project"
    exit 1
fi

STATUS_FIELD=$(gh project field-list "$PROJECT_NUM" --owner "$OWNER" --format json | jq -r '.fields[] | select(.name=="Status") | .id')
STATUS_OPTION=$(get_status_option)

if [ -z "$STATUS_OPTION" ]; then
    echo "Error: Could not find status option for '$STATUS'"
    echo "Available options:"
    gh project field-list "$PROJECT_NUM" --owner "$OWNER" --format json | jq '.fields[] | select(.name=="Status") | .options'
    exit 1
fi

gh project item-edit --project-id "$PROJECT_NUM" --id "$ITEM_ID" --field-id "$STATUS_FIELD" --single-select-option-id "$STATUS_OPTION"

echo "✓ Updated $FEATURE_ID → $STATUS"
