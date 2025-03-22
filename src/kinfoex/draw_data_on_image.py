import math

import PIL.Image
import PIL.ImageDraw

BLACK = "#000000"
WHITE = "#FFFFFF"


def get_labels(data):
    labels = set()
    for d in data:
        label = d["label"]
        labels.add(label)
    return list(labels)


def box_mid_point(box):
    x = (box[0] + box[2]) // 2
    y = (box[1] + box[3]) // 2
    return [x, y]


def hex_to_rgb(hex_color, alpha_percent=None):
    """
    Convert a hexadecimal color to an RGB or RGBA tuple.

    Parameters:
    hex_color (str): The hexadecimal color code (e.g., "#000000").
    alpha_percent (float, optional): The alpha value as a percentage (0-100). Defaults to None.

    Returns:
    tuple: A tuple representing the RGB or RGBA color.
    """
    # Remove the hash symbol if present
    hex_color = hex_color.lstrip("#")

    # Convert hex to RGB
    rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    # Convert alpha percentage to a value between 0 and 255
    if alpha_percent is not None:
        alpha = int((alpha_percent / 100) * 255)
        return rgb + (alpha,)
    else:
        return rgb


def draw_arrow_on_image(
    image_draw, line, outline_width=4, color=(0, 0, 0), outline_color=None
):

    head_size = round(outline_width * 3)
    head_width = round(outline_width * 7)

    x = 1 - head_width / line_length(line)

    x0, y0 = line[0]
    x1, y1 = line[1]

    xb = x * (x1 - x0) + x0
    yb = x * (y1 - y0) + y0

    alpha = math.atan2(y1 - y0, x1 - x0) - 90.0 * math.pi / 180.0
    a = head_size * math.cos(alpha)
    b = head_size * math.sin(alpha)
    vtx0 = (xb + a, yb + b)
    vtx1 = (xb - a, yb - b)

    image_draw.polygon(
        [vtx0, vtx1, line[1]],
        fill=color,
        outline=outline_color,
        width=outline_width,
    )

    base_line = shorten_line(line, x=x)

    if outline_color is not None:
        image_draw.line(base_line, width=outline_width, fill=outline_color)
    image_draw.line(base_line, width=outline_width * 2, fill=color)


def line_length(line):
    x0, y0 = line[0]
    x1, y1 = line[1]
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)


def shorten_line(line, x):
    x0, y0 = line[0]
    x1, y1 = line[1]
    dx = x1 - x0
    dy = y1 - y0
    return [(x0, y0), (x0 + x * dx, y0 + x * dy)]


def draw_data_on_image(
    image,
    data,
    label2color=None,
    draw_links=True,
    labels_color=True,
    links_color=True,
    box_outline_width=2,
    box_outline_color="color",
    fill_color_alpha=40.0,
    links_color_alpha=75.0,
    draw_arrow=True,
):
    # line
    box_outline_width = round(
        (box_outline_width / 1000000) * image.size[0] * image.size[1]
    )
    # image
    image = image.copy().convert("RGBA")
    image_boxes = PIL.Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = PIL.ImageDraw.Draw(image_boxes)
    # label2color
    if label2color is None:
        label2color = {}
        labels = get_labels(data)
        for i, label in enumerate(labels):
            label2color[label] = BLACK
    label2color[""] = BLACK
    #
    boxes_per_color = {"white": [], "black": []}
    for color in label2color.values():
        boxes_per_color[color] = []
    for i in range(len(data)):
        box = data[i]["box"]
        id = data[i]["id"]
        links = data[i]["links"]
        label = data[i]["label"]
        if not labels_color:
            color = "white"
        else:
            color = label2color[label]
        boxes_per_color[color].append(box)
    for color in boxes_per_color.keys():
        for box in boxes_per_color[color]:
            x = box_outline_width
            box = [box[0] - x, box[1] - x, box[2] + x, box[3] + x]
            fill = hex_to_rgb(color, fill_color_alpha)
            outline = hex_to_rgb(color) if box_outline_color == "color" else None
            draw.rectangle(
                box,
                fill=fill,
                outline=outline,
                width=box_outline_width,
            )
    if draw_links:
        links = []
        for entity in data:
            for link in entity["links"]:
                if link not in links:
                    links.append(link)
        links = sorted(links, key=lambda link: link[0])
        labels = []
        lines = []
        for link in links:
            p_1, p_2 = (0, 0), (0, 0)
            color = (0, 0, 0)
            for entity in data:
                if entity["id"] == link[0]:
                    p_1 = box_mid_point(entity["box"])
                    label = entity["label"]
                    for entity in data:
                        if entity["id"] == link[1]:
                            p_2 = box_mid_point(entity["box"])
                            line = (tuple(p_1), tuple(p_2))
                            labels.append(label)
                            lines.append(line)
                            break
                    break
        l = list(label2color.keys())
        l.reverse()
        for label in l:
            if links_color:
                color = label2color[label]
            else:
                color = BLACK
            color = hex_to_rgb(color, links_color_alpha)
            outline = color if box_outline_color == "color" else hex_to_rgb(BLACK)
            for line_label, line in zip(labels, lines):
                if line_label == label:
                    if draw_arrow:
                        draw_arrow_on_image(
                            image_draw=draw,
                            line=line,
                            outline_width=box_outline_width,
                            color=color,
                            outline_color=outline,
                        )
                    else:
                        draw.line(line, fill=outline, width=4 * box_outline_width)
                        draw.line(line, fill=color, width=3 * box_outline_width)
    image_boxes = PIL.Image.alpha_composite(image, image_boxes)
    return image_boxes
