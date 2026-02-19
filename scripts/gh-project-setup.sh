#!/bin/bash
# Setup GitHub Project for cascade-rs
# Run after: gh auth refresh -s project -h github.com

set -e

OWNER="DonSqualo"
REPO="cascade-rs"

echo "Creating GitHub Project..."
PROJECT_URL=$(gh project create --owner "$OWNER" --title "cascade-rs Feature Parity" --format json | jq -r '.url')
PROJECT_NUM=$(echo "$PROJECT_URL" | grep -oP '/projects/\K\d+')

echo "Project created: $PROJECT_URL (number: $PROJECT_NUM)"

# Link to repo
echo "Linking project to repo..."
gh project link "$PROJECT_NUM" --owner "$OWNER" --repo "$OWNER/$REPO"

# Get the Status field ID (auto-created)
echo "Getting Status field..."
STATUS_FIELD=$(gh project field-list "$PROJECT_NUM" --owner "$OWNER" --format json | jq -r '.fields[] | select(.name=="Status") | .id')

echo "Status field ID: $STATUS_FIELD"

# Create items for all 44 features
echo "Creating feature items..."

# Primitives
for feature in "primitive::box|Box/cuboid creation" "primitive::sphere|Sphere creation" "primitive::cylinder|Cylinder creation" "primitive::cone|Cone creation" "primitive::torus|Torus creation" "primitive::wedge|Wedge/prism creation"; do
    id=$(echo "$feature" | cut -d'|' -f1)
    name=$(echo "$feature" | cut -d'|' -f2)
    gh project item-create "$PROJECT_NUM" --owner "$OWNER" --title "$id" --body "$name" 2>/dev/null || true
    echo "  Created: $id"
done

# Boolean Operations
for feature in "boolean::fuse|Union of solids" "boolean::cut|Difference of solids" "boolean::common|Intersection of solids" "boolean::section|Section (solid/plane)"; do
    id=$(echo "$feature" | cut -d'|' -f1)
    name=$(echo "$feature" | cut -d'|' -f2)
    gh project item-create "$PROJECT_NUM" --owner "$OWNER" --title "$id" --body "$name" 2>/dev/null || true
    echo "  Created: $id"
done

# BREP Core
for feature in "brep::vertex|Vertex representation" "brep::edge|Edge representation" "brep::wire|Wire (connected edges)" "brep::face|Face representation" "brep::shell|Shell (connected faces)" "brep::solid|Solid representation" "brep::compound|Compound shapes" "brep::topology|Topological queries"; do
    id=$(echo "$feature" | cut -d'|' -f1)
    name=$(echo "$feature" | cut -d'|' -f2)
    gh project item-create "$PROJECT_NUM" --owner "$OWNER" --title "$id" --body "$name" 2>/dev/null || true
    echo "  Created: $id"
done

# Curves
for feature in "curve::line|Line segment" "curve::circle|Circle/arc" "curve::ellipse|Ellipse" "curve::bspline|B-spline curve" "curve::bezier|Bezier curve"; do
    id=$(echo "$feature" | cut -d'|' -f1)
    name=$(echo "$feature" | cut -d'|' -f2)
    gh project item-create "$PROJECT_NUM" --owner "$OWNER" --title "$id" --body "$name" 2>/dev/null || true
    echo "  Created: $id"
done

# Surfaces
for feature in "surface::plane|Planar surface" "surface::cylinder|Cylindrical surface" "surface::sphere|Spherical surface" "surface::bspline|B-spline surface" "surface::bezier|Bezier surface"; do
    id=$(echo "$feature" | cut -d'|' -f1)
    name=$(echo "$feature" | cut -d'|' -f2)
    gh project item-create "$PROJECT_NUM" --owner "$OWNER" --title "$id" --body "$name" 2>/dev/null || true
    echo "  Created: $id"
done

# Modifications
for feature in "modify::fillet|Edge filleting" "modify::chamfer|Edge chamfering" "modify::offset|Solid offset/shell" "modify::transform|Affine transforms"; do
    id=$(echo "$feature" | cut -d'|' -f1)
    name=$(echo "$feature" | cut -d'|' -f2)
    gh project item-create "$PROJECT_NUM" --owner "$OWNER" --title "$id" --body "$name" 2>/dev/null || true
    echo "  Created: $id"
done

# Meshing
for feature in "mesh::triangulate|Surface triangulation" "mesh::quality|Mesh quality control" "mesh::export_stl|STL export"; do
    id=$(echo "$feature" | cut -d'|' -f1)
    name=$(echo "$feature" | cut -d'|' -f2)
    gh project item-create "$PROJECT_NUM" --owner "$OWNER" --title "$id" --body "$name" 2>/dev/null || true
    echo "  Created: $id"
done

# File I/O
for feature in "io::step_read|STEP file import" "io::step_write|STEP file export" "io::brep_read|Native BREP import" "io::brep_write|Native BREP export"; do
    id=$(echo "$feature" | cut -d'|' -f1)
    name=$(echo "$feature" | cut -d'|' -f2)
    gh project item-create "$PROJECT_NUM" --owner "$OWNER" --title "$id" --body "$name" 2>/dev/null || true
    echo "  Created: $id"
done

# Geometric Queries
for feature in "query::distance|Min distance between shapes" "query::intersection|Shape intersection test" "query::inside|Point-in-solid test" "query::bounds|Bounding box" "query::properties|Volume, area, center of mass"; do
    id=$(echo "$feature" | cut -d'|' -f1)
    name=$(echo "$feature" | cut -d'|' -f2)
    gh project item-create "$PROJECT_NUM" --owner "$OWNER" --title "$id" --body "$name" 2>/dev/null || true
    echo "  Created: $id"
done

echo ""
echo "✓ Project setup complete!"
echo "  URL: $PROJECT_URL"
echo "  Project number: $PROJECT_NUM"
echo ""
echo "Save this for scripts/gh-project-update.sh:"
echo "  PROJECT_NUM=$PROJECT_NUM"
echo "  STATUS_FIELD=$STATUS_FIELD"
