"""
P&ID Dataset Visualization Demo
Quick standalone script to visualize P&ID annotations without training
"""

from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class PIDVisualizer:
    """
    Standalone P&ID annotation visualizer
    """

    def __init__(self, data_dir="dataset", images_dir="images"):
        self.data_dir = Path(data_dir)
        self.images_dir = Path(images_dir)

        # Define class colors and names
        self.class_names = {
            "symbols": "Symbol",
            "words": "Word/Text",
            "lines": "Line",
            "lines2": "Line2",
            "Table": "Table",
            "KeyValue": "KeyValue",
        }

        self.colors = {
            "symbols": "#FF6B6B",  # Red
            "words": "#4ECDC4",  # Teal
            "lines": "#45B7D1",  # Blue
            "lines2": "#96CEB4",  # Green
            "Table": "#FFEAA7",  # Yellow
            "KeyValue": "#DDA0DD",  # Plum
        }

        # Find available data indices
        self.available_indices = self.find_available_data()
        print(f"Found {len(self.available_indices)} P&ID diagrams")

    def find_available_data(self):
        """Find all available data indices"""
        indices = []
        for symbols_file in self.data_dir.glob("*/*_symbols.npy"):
            idx = symbols_file.parent.name
            indices.append(idx)
        return sorted(indices)

    def load_annotations(self, idx):
        """Load all annotation types for a given index"""
        annotations = {}
        data_path = self.data_dir / str(idx)

        annotation_types = [
            "KeyValue",
            "lines",
            "lines2",
            "linker",
            "symbols",
            "Table",
            "words",
        ]

        for ann_type in annotation_types:
            file_path = data_path / f"{idx}_{ann_type}.npy"
            if file_path.exists():
                try:
                    data = np.load(file_path, allow_pickle=True)
                    annotations[ann_type] = data
                    print(f"  {ann_type}: {len(data)} items")
                except Exception as e:
                    print(f"  Error loading {ann_type}: {e}")
                    annotations[ann_type] = np.array([])
            else:
                annotations[ann_type] = np.array([])

        return annotations

    def load_image(self, idx):
        """Load P&ID image"""
        # Try different extensions
        for ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            image_path = self.images_dir / f"{idx}{ext}"
            if image_path.exists():
                image = cv2.imread(str(image_path))
                if image is not None:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print(f"Warning: Image {idx} not found")
        return None

    def parse_bbox(self, bbox_data):
        """Parse bounding box from various formats"""
        try:
            if isinstance(bbox_data, (list, np.ndarray)):
                bbox = list(bbox_data)
                if len(bbox) >= 4:
                    return [float(x) for x in bbox[:4]]
            return None
        except:
            return None

    def parse_line_coords(self, line_data):
        """Parse line coordinates from lines data"""
        try:
            if isinstance(line_data, (list, np.ndarray)) and len(line_data) >= 4:
                coords = [float(x) for x in line_data[:4]]
                return coords
            return None
        except:
            return None

    def visualize_sample(self, idx=None, save_path=None, show_details=True):
        """
        Visualize annotations for a specific P&ID sample
        """
        if idx is None:
            idx = self.available_indices[0] if self.available_indices else "2"

        if isinstance(idx, int):
            idx = str(idx)

        print(f"\n=== Visualizing P&ID Sample {idx} ===")

        # Load image
        image = self.load_image(idx)
        if image is None:
            # Create dummy image
            image = np.ones((1000, 1000, 3), dtype=np.uint8) * 240
            print("Using dummy image")

        # Load annotations
        print("Loading annotations:")
        annotations = self.load_annotations(idx)

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.imshow(image)

        annotation_count = 0

        # Process each annotation type
        for ann_type, ann_data in annotations.items():
            if len(ann_data) == 0:
                continue

            color = self.colors.get(ann_type, "#888888")
            display_name = self.class_names.get(ann_type, ann_type)

            if ann_type == "symbols":
                annotation_count += self.draw_symbols(ax, ann_data, color, show_details)
            elif ann_type == "words":
                annotation_count += self.draw_words(ax, ann_data, color, show_details)
            elif ann_type in ["lines", "lines2"]:
                annotation_count += self.draw_lines(
                    ax, ann_data, color, ann_type, show_details
                )
            elif ann_type == "Table":
                annotation_count += self.draw_tables(ax, ann_data, color, show_details)
            elif ann_type == "KeyValue":
                annotation_count += self.draw_keyvalues(
                    ax, ann_data, color, show_details
                )

        # Set title and styling
        ax.set_title(
            f"P&ID Annotations - Sample {idx}\n{annotation_count} total annotations",
            fontsize=16,
            weight="bold",
            pad=20,
        )
        ax.axis("off")

        # Create legend
        legend_elements = []
        for ann_type, color in self.colors.items():
            if ann_type in annotations and len(annotations[ann_type]) > 0:
                count = len(annotations[ann_type])
                display_name = self.class_names.get(ann_type, ann_type)
                legend_elements.append(
                    patches.Patch(
                        color=color, alpha=0.7, label=f"{display_name} ({count})"
                    )
                )

        if legend_elements:
            ax.legend(
                handles=legend_elements,
                loc="upper left",
                bbox_to_anchor=(0, 1),
                framealpha=0.9,
                fontsize=12,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"Visualization saved to {save_path}")

        plt.show()
        return fig

    def draw_symbols(self, ax, symbols_data, color, show_details=True):
        """Draw symbol annotations"""
        count = 0
        for i, symbol_data in enumerate(symbols_data):
            if len(symbol_data) >= 2:
                symbol_id = symbol_data[0] if len(symbol_data) > 0 else f"symbol_{i}"
                bbox = self.parse_bbox(symbol_data[1])
                symbol_type = symbol_data[2] if len(symbol_data) > 2 else ""

                if bbox:
                    x1, y1, x2, y2 = bbox
                    width = x2 - x1
                    height = y2 - y1

                    # Draw rectangle
                    rect = patches.Rectangle(
                        (x1, y1),
                        width,
                        height,
                        linewidth=2.5,
                        edgecolor=color,
                        facecolor="none",
                        alpha=0.8,
                    )
                    ax.add_patch(rect)

                    # Add label
                    if show_details:
                        label = f"S{i + 1}"
                        if symbol_type:
                            label += f" ({symbol_type})"

                        ax.text(
                            x1,
                            y1 - 8,
                            label,
                            bbox=dict(
                                boxstyle="round,pad=0.3", facecolor=color, alpha=0.8
                            ),
                            fontsize=9,
                            color="white",
                            weight="bold",
                        )

                    count += 1
        return count

    def draw_words(self, ax, words_data, color, show_details=True):
        """Draw word/text annotations"""
        count = 0
        for i, word_data in enumerate(words_data):
            if len(word_data) >= 2:
                word_id = word_data[0] if len(word_data) > 0 else f"word_{i}"
                bbox = self.parse_bbox(word_data[1])
                text_content = word_data[2] if len(word_data) > 2 else ""

                if bbox:
                    x1, y1, x2, y2 = bbox
                    width = x2 - x1
                    height = y2 - y1

                    # Draw rectangle
                    rect = patches.Rectangle(
                        (x1, y1),
                        width,
                        height,
                        linewidth=2,
                        edgecolor=color,
                        facecolor=color,
                        alpha=0.3,
                    )
                    ax.add_patch(rect)

                    # Add text label
                    if show_details and text_content:
                        # Truncate long text
                        display_text = (
                            str(text_content)[:15] + "..."
                            if len(str(text_content)) > 15
                            else str(text_content)
                        )

                        ax.text(
                            x1,
                            y1 - 5,
                            display_text,
                            bbox=dict(
                                boxstyle="round,pad=0.2", facecolor=color, alpha=0.9
                            ),
                            fontsize=8,
                            color="white",
                            weight="bold",
                        )

                    count += 1
        return count

    def draw_lines(self, ax, lines_data, color, line_type, show_details=True):
        """Draw line annotations"""
        count = 0
        for i, line_data in enumerate(lines_data):
            try:
                if line_type == "lines" and len(line_data) >= 2:
                    # Format: ['line_1', [x1, y1, x2, y2], '', 'solid']
                    coords = self.parse_line_coords(line_data[1])
                    line_style = line_data[3] if len(line_data) > 3 else "solid"
                elif line_type == "lines2" and len(line_data) >= 4:
                    # Format: [x1, y1, x2, y2, type]
                    coords = [float(x) for x in line_data[:4]]
                    line_style = "solid"
                else:
                    continue

                if coords and len(coords) >= 4:
                    x1, y1, x2, y2 = coords

                    # Draw line
                    linestyle = "--" if "dash" in str(line_style).lower() else "-"
                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color=color,
                        linewidth=3,
                        alpha=0.8,
                        linestyle=linestyle,
                    )

                    # Add small markers at endpoints
                    ax.plot(
                        [x1, x2], [y1, y2], "o", color=color, markersize=4, alpha=0.8
                    )

                    # Add line ID if showing details
                    if show_details:
                        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                        ax.text(
                            mid_x,
                            mid_y,
                            f"L{i + 1}",
                            bbox=dict(
                                boxstyle="round,pad=0.2", facecolor=color, alpha=0.7
                            ),
                            fontsize=7,
                            color="white",
                            weight="bold",
                            ha="center",
                            va="center",
                        )

                    count += 1
            except Exception as e:
                if show_details:
                    print(f"  Error drawing line {i}: {e}")
                continue

        return count

    def draw_tables(self, ax, table_data, color, show_details=True):
        """Draw table annotations (usually headers)"""
        count = 0
        if len(table_data) > 0:
            # Tables are usually header rows - draw as text boxes
            for i, row in enumerate(table_data):
                if isinstance(row, (list, np.ndarray)) and len(row) > 0:
                    # Create a simple representation
                    y_pos = 50 + i * 25  # Position near top
                    text = " | ".join(
                        str(cell) for cell in row[:4]
                    )  # Show first 4 columns

                    ax.text(
                        50,
                        y_pos,
                        f"Table Row {i + 1}: {text}",
                        bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8),
                        fontsize=10,
                        color="white",
                        weight="bold",
                    )
                    count += 1

        return count

    def draw_keyvalues(self, ax, keyvalue_data, color, show_details=True):
        """Draw key-value pair annotations"""
        count = 0
        for i, kv_data in enumerate(keyvalue_data):
            if isinstance(kv_data, (list, np.ndarray)) and len(kv_data) >= 2:
                key = str(kv_data[0])
                value = str(kv_data[1])

                # Position key-value pairs on the image
                y_pos = 50 + (i * 30)
                x_pos = ax.get_xlim()[1] - 300  # Right side of image

                ax.text(
                    x_pos,
                    y_pos,
                    f"{key}: {value}",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.8),
                    fontsize=9,
                    color="white",
                    weight="bold",
                )
                count += 1

        return count

    def create_overview(self, num_samples=5, save_dir="visualizations"):
        """Create overview visualizations for multiple samples"""
        print(f"\n=== Creating Overview of {num_samples} Samples ===")

        # Create output directory
        if save_dir:
            Path(save_dir).mkdir(exist_ok=True)

        samples_to_show = self.available_indices[:num_samples]

        for i, idx in enumerate(samples_to_show):
            print(f"\nProcessing sample {i + 1}/{len(samples_to_show)}: {idx}")

            save_path = None
            if save_dir:
                save_path = Path(save_dir) / f"pid_sample_{idx}_annotated.png"

            try:
                self.visualize_sample(idx=idx, save_path=save_path, show_details=True)
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")

        print(f"\nOverview complete! Visualizations saved to '{save_dir}' directory")

    def analyze_dataset(self):
        """Analyze the dataset and show statistics"""
        print("\n=== Dataset Analysis ===")

        stats = {
            "total_samples": len(self.available_indices),
            "annotation_types": {},
            "total_annotations": 0,
        }

        for idx in self.available_indices:
            try:
                annotations = self.load_annotations(idx)
                for ann_type, ann_data in annotations.items():
                    if ann_type not in stats["annotation_types"]:
                        stats["annotation_types"][ann_type] = 0
                    stats["annotation_types"][ann_type] += len(ann_data)
                    stats["total_annotations"] += len(ann_data)
            except Exception as e:
                print(f"Error analyzing sample {idx}: {e}")
                continue

        # Print statistics
        print(f"Total P&ID samples: {stats['total_samples']}")
        print(f"Total annotations: {stats['total_annotations']}")
        print("\nAnnotation breakdown:")
        for ann_type, count in stats["annotation_types"].items():
            percentage = (
                (count / stats["total_annotations"] * 100)
                if stats["total_annotations"] > 0
                else 0
            )
            display_name = self.class_names.get(ann_type, ann_type)
            print(f"  {display_name}: {count} ({percentage:.1f}%)")

        # Create bar chart
        if stats["annotation_types"]:
            plt.figure(figsize=(12, 8))

            types = list(stats["annotation_types"].keys())
            counts = list(stats["annotation_types"].values())
            colors_list = [self.colors.get(t, "#888888") for t in types]

            bars = plt.bar(range(len(types)), counts, color=colors_list, alpha=0.8)
            plt.xlabel("Annotation Types")
            plt.ylabel("Count")
            plt.title("P&ID Dataset - Annotation Type Distribution")
            plt.xticks(
                range(len(types)),
                [self.class_names.get(t, t) for t in types],
                rotation=45,
            )

            # Add count labels on bars
            for bar, count in zip(bars, counts):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts) * 0.01,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            plt.tight_layout()
            plt.savefig("dataset_analysis.png", dpi=150, bbox_inches="tight")
            plt.show()

        return stats


def main():
    """Main demo function"""
    print("=== P&ID Dataset Visualization Demo ===")

    # Initialize visualizer
    visualizer = PIDVisualizer(data_dir="dataset", images_dir="images")

    if not visualizer.available_indices:
        print("No P&ID data found! Please check your dataset and images directories.")
        print("Expected structure:")
        print("  dataset/")
        print("    2/")
        print("      2_symbols.npy")
        print("      2_words.npy")
        print("      ...")
        print("  images/")
        print("    2.png (or 2.jpg)")
        return

    # Analyze dataset
    stats = visualizer.analyze_dataset()

    # Show first sample in detail
    print(f"\n=== Detailed View of Sample {visualizer.available_indices[0]} ===")
    visualizer.visualize_sample(
        idx=visualizer.available_indices[0],
        save_path="sample_detailed.png",
        show_details=True,
    )

    # Create overview of multiple samples
    visualizer.create_overview(
        num_samples=min(3, len(visualizer.available_indices)),
        save_dir="sample_visualizations",
    )

    print("\n=== Demo Complete! ===")
    print("Generated files:")
    print("  - dataset_analysis.png (annotation statistics)")
    print("  - sample_detailed.png (detailed view)")
    print("  - sample_visualizations/ (directory with multiple samples)")


if __name__ == "__main__":
    main()
