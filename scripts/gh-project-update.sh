#!/bin/bash
# Update cascade-rs GitHub Project item status
# Usage: ./gh-project-update.sh <feature_id> <status>
# Status: backlog | in-progress | verify | done

set -e

FEATURE_ID="$1"
STATUS="$2"
OWNER="DonSqualo"
PROJECT_NUM="1"
PROJECT_ID="PVT_kwHOBWweZs4BPktV"

if [ -z "$FEATURE_ID" ] || [ -z "$STATUS" ]; then
    echo "Usage: $0 <feature_id> <status>"
    echo "  feature_id: e.g., primitive::box"
    echo "  status: backlog | in-progress | verify | done"
    exit 1
fi

# Map status to option name
case "$STATUS" in
    backlog)     STATUS_NAME="Todo" ;;
    in-progress) STATUS_NAME="In Progress" ;;
    verify)      STATUS_NAME="In Review" ;;
    done)        STATUS_NAME="Done" ;;
    *)           echo "Unknown status: $STATUS"; exit 1 ;;
esac

echo "Updating $FEATURE_ID to $STATUS ($STATUS_NAME)..."

# Get item ID by title
ITEM_ID=$(gh project item-list "$PROJECT_NUM" --owner "$OWNER" --format json | \
    jq -r ".items[] | select(.content.title == \"$FEATURE_ID\") | .id")

if [ -z "$ITEM_ID" ]; then
    echo "Error: Item '$FEATURE_ID' not found in project"
    exit 1
fi

echo "  Item ID: $ITEM_ID"

# Get Status field ID and option ID
FIELD_DATA=$(gh project field-list "$PROJECT_NUM" --owner "$OWNER" --format json)
STATUS_FIELD=$(echo "$FIELD_DATA" | jq -r '.fields[] | select(.name=="Status") | .id')
STATUS_OPTION=$(echo "$FIELD_DATA" | jq -r ".fields[] | select(.name==\"Status\") | .options[] | select(.name==\"$STATUS_NAME\") | .id")

if [ -z "$STATUS_OPTION" ]; then
    echo "Error: Could not find status option for '$STATUS_NAME'"
    echo "Available options:"
    echo "$FIELD_DATA" | jq '.fields[] | select(.name=="Status") | .options[].name'
    exit 1
fi

echo "  Status field: $STATUS_FIELD"
echo "  Option ID: $STATUS_OPTION"

# Update the item
gh project item-edit --project-id "$PROJECT_ID" --id "$ITEM_ID" --field-id "$STATUS_FIELD" --single-select-option-id "$STATUS_OPTION"

echo "✓ Updated $FEATURE_ID → $STATUS"
