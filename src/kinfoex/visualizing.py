import math
import PIL.Image
import PIL.ImageDraw

BOX_NORMILIZER = 1000

COLORS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "scarlet": (222, 31, 38),
    "yellowish_orange": (246, 168, 0),
    "azul_ufla": (0, 75, 128),  # Azul-UFLA
    "verde_ufla": (0, 148, 62),  # Verde-UFLA
}


def unnormalize_box(
    box: list[int], image_size: tuple[int, int], box_max_size: int
) -> list[int]:
    return [
        int(image_size[0] * (box[0] / box_max_size)),
        int(image_size[1] * (box[1] / box_max_size)),
        int(image_size[0] * (box[2] / box_max_size)),
        int(image_size[1] * (box[3] / box_max_size)),
    ]


def unnormalize_boxes(
    boxes: list[list[int]], image_size: tuple[int, int], box_max_size
) -> list[list[int]]:
    return [unnormalize_box(box, image_size, box_max_size) for box in boxes]


def add_alpha(color, alpha):
    color = list(color)
    color.append(alpha)
    return tuple(color)


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


def draw_arrow_on_image(image_draw, line, width=1, color=(0, 0, 0), outline_color=None):

    head_size = round(width * 2)
    head_width = round(width * 5)

    length = line_length(line)
    if length == 0:
        return

    x = 1 - head_width / length

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
        width=round(0.3 * width),
    )

    base_line = shorten_line(line, x=x)

    if outline_color is not None:
        image_draw.line(base_line, width=round(width * 1.5), fill=outline_color)
    image_draw.line(base_line, width=width, fill=color)


def join_boxes(boxes_to_join):
    box = [0, 0, 0, 0]
    if not boxes_to_join:
        return box
    x0_list = [box[0] for box in boxes_to_join]
    y0_list = [box[1] for box in boxes_to_join]
    x1_list = [box[2] for box in boxes_to_join]
    y2_list = [box[3] for box in boxes_to_join]
    box = [min(x0_list), min(y0_list), max(x1_list), max(y2_list)]
    return box


def box_mid_point(box):
    x = (box[0] + box[2]) // 2
    y = (box[1] + box[3]) // 2
    return [x, y]


def create_entities_from_labels(labels, ner_id2label, re_label2id):
    entities = {"start": [], "end": [], "label": []}
    current_label = None

    for i, label_id in enumerate(labels):
        if label_id not in ner_id2label:
            continue
        iob_label = ner_id2label[label_id]
        if iob_label == "O":
            if current_label is not None:
                entities["end"].append(i)
                current_label = None
            continue
        iob_tag, label = iob_label.split("-")
        if iob_tag == "B":
            if current_label is not None:
                entities["end"].append(i)
                current_label = None
            entities["start"].append(i)
            entities["label"].append(re_label2id[label])
            current_label = label
        elif label.startswith("I"):
            if current_label is not None and label != current_label:
                entities["end"].append(i)
                current_label = None
            else:
                entities["start"].append(i)
                entities["label"].append(re_label2id[label])
                current_label = label
    if current_label is not None:
        entities["end"].append(len(labels))
        current_label = None

    return entities


def draw_sample(
    sample,
    label2color,
    draw_entities=False,
    ner_id2label=None,
    re_label2id=None,
    is_bbox_normalized=True,
    draw_relations=True,
    ground_truth_sample=None,
):

    # image
    image = sample["original_image"].copy().convert("RGBA")
    image_boxes = PIL.Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = PIL.ImageDraw.Draw(image_boxes)

    # line
    line_width = round((1 / 500) * image.size[0])

    input_ids = sample["input_ids"]
    bbox = sample["bbox"]
    labels = sample["labels"]

    if is_bbox_normalized:
        bbox = unnormalize_boxes(bbox, image.size, box_max_size=BOX_NORMILIZER)

    entities = None
    if draw_entities:
        assert ner_id2label is not None and re_label2id is not None
        boxes = []
        box_label = []
        for box, label in zip(bbox, labels):
            if label in ner_id2label:
                if ner_id2label[label] == "O":
                    boxes.append(box)
                    box_label.append(label)
        entities = create_entities_from_labels(
            labels, ner_id2label=ner_id2label, re_label2id=re_label2id
        )
        for start, end in zip(entities["start"], entities["end"]):
            box = join_boxes(bbox[start:end])
            boxes.append(box)
            label = labels[start]
            box_label.append(label)
    else:
        boxes = bbox
        box_label = labels

    for box, label in zip(boxes, box_label):
        if label not in label2color:
            continue
        color = label2color[label]
        fill = add_alpha(color, 96)
        draw.rectangle(
            xy=(box[0], box[1], box[2], box[3]),
            fill=fill,
            outline="black",
            width=round(60 / 100 * line_width),
        )

    if draw_entities and draw_relations:

        arrows_for_draw = {"line": [], "color": []}

        true_positives = []

        if ground_truth_sample is not None:

            for head, tail, label in zip(
                ground_truth_sample["relations"]["head"],
                ground_truth_sample["relations"]["tail"],
                ground_truth_sample["relations"]["label"],
            ):

                if label == 0:
                    continue

                k_start, k_end = (
                    ground_truth_sample["entities"]["start"][head],
                    ground_truth_sample["entities"]["end"][head],
                )
                v_start, v_end = (
                    ground_truth_sample["entities"]["start"][tail],
                    ground_truth_sample["entities"]["end"][tail],
                )
                k_box = join_boxes(bbox[k_start:k_end])
                v_box = join_boxes(bbox[v_start:v_end])
                k_point = box_mid_point(k_box)
                v_point = box_mid_point(v_box)
                line = (tuple(k_point), tuple(v_point))
                color = COLORS["white"]
                true_positives.append(line)
                arrows_for_draw["line"].append(line)
                arrows_for_draw["color"].append(color)

        for head, tail, label in zip(
            sample["relations"]["head"],
            sample["relations"]["tail"],
            sample["relations"]["label"],
        ):
            if label == 0:
                continue

            if len(sample["entities"]["start"]) <= head:
                break
            k_start, k_end = (
                sample["entities"]["start"][head],
                sample["entities"]["end"][head],
            )
            v_start, v_end = (
                sample["entities"]["start"][tail],
                sample["entities"]["end"][tail],
            )
            k_box = join_boxes(bbox[k_start:k_end])
            v_box = join_boxes(bbox[v_start:v_end])
            k_point = box_mid_point(k_box)
            v_point = box_mid_point(v_box)

            line = (tuple(k_point), tuple(v_point))
            label = labels[k_start]
            if label not in label2color:
                color = COLORS["scarlet"]
            else:
                color = label2color[label]
            if ground_truth_sample is not None:
                if line not in true_positives:
                    color = COLORS["scarlet"]
            color = add_alpha(color, 96)

            arrows_for_draw["line"].append(line)
            arrows_for_draw["color"].append(color)

        for line, color in zip(arrows_for_draw["line"], arrows_for_draw["color"]):
            draw_arrow_on_image(
                draw,
                line=line,
                color=color,
                width=line_width,
                outline_color=COLORS["black"],
            )

    image_boxes = PIL.Image.alpha_composite(image, image_boxes)

    return image_boxes


def entities_table(sample, tokenizer=None):
    entities = {
        "location": [
            (start, end)
            for start, end in zip(
                sample["entities"]["start"], sample["entities"]["end"]
            )
        ],
        "text": [
            sample["input_ids"][start:end]
            for start, end in zip(
                sample["entities"]["start"], sample["entities"]["end"]
            )
        ],
        "label": [label for label in sample["entities"]["label"]],
    }
    if tokenizer:
        entities["text"] = [
            tokenizer.decode(entitie_text) for entitie_text in entities["text"]
        ]
    return entities


def key_value_table(sample, tokenizer=None):
    entities = entities_table(sample, tokenizer)
    key_value = {
        "key location": [
            entities["location"][head] for head in sample["relations"]["head"]
        ],
        "key": [entities["text"][head] for head in sample["relations"]["head"]],
        "key label": [entities["label"][head] for head in sample["relations"]["head"]],
        "value location": [
            entities["location"][tail] for tail in sample["relations"]["tail"]
        ],
        "value": [entities["text"][tail] for tail in sample["relations"]["tail"]],
        "value label": [
            entities["label"][tail] for tail in sample["relations"]["tail"]
        ],
    }
    return key_value
