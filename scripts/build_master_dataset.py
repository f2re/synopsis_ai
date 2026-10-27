#!/usr/bin/env python3
"""Builds the master high-fidelity training dataset from user-provided canonical reference art.
Extracts semantic crops, portraits, equipment, interiors, and atmospheric weather scenes
with detailed Russian synoptic school captions.
"""
from __future__ import annotations

import os
from pathlib import Path
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
REF_DIR = REPO_ROOT / "design/pixel-art/references/master_26063"
OUT_DIR = REPO_ROOT / "datasets/master_style_lora"


def build_dataset() -> list[tuple[str, Image.Image, str]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples: list[tuple[str, Image.Image, str]] = []

    # 1. Office Interior
    p1 = REF_DIR / "ref_office_interior_panoramic.jpg"
    if p1.is_file():
        img1 = Image.open(p1)
        samples.append((
            "office_interior_full",
            img1.resize((512, 288), Image.Resampling.LANCZOS),
            "<26063_pixel_style>, synoptic office interior, panoramic window overlooking Gulf of Finland, St Petersburg skyline in distance, wooden workstation desk, dual computer monitors displaying isobar weather maps and satellite cloud loops, meteorological barometers and anemometers, cozy warm lighting, birch and pine trees outside, modern high quality pixel art"
        ))
        samples.append((
            "office_desk_monitors",
            img1.crop((550, 200, 1024, 576)).resize((512, 512), Image.Resampling.LANCZOS),
            "<26063_pixel_style>, synoptic workstation desk, dual monitor setup displaying Russian synoptic surface charts with blue and red isobars and satellite meteorology, open weather journal logbook, coffee mug, green desk lamp, indoor plants, detailed pixel art interior"
        ))
        samples.append((
            "office_window_instruments",
            img1.crop((100, 50, 600, 550)).resize((512, 512), Image.Resampling.LANCZOS),
            "<26063_pixel_style>, meteorological instruments on window sill, anemometer, thermometer shelter model, barometer, view of Gulf of Finland and Peter and Paul fortress spire across water, birch and pine branches, fluffy cumulus clouds, peaceful northern summer atmosphere, crisp pixel art"
        ))
        samples.append((
            "office_synoptic_map_wall",
            img1.crop((700, 50, 1000, 350)).resize((512, 512), Image.Resampling.LANCZOS),
            "<26063_pixel_style>, wall-mounted synoptic meteorology charts, northern Europe weather cyclone and anticyclone isobar maps, bookshelf with atmospheric science books, warm hanging lamp, cozy synoptics room pixel art"
        ))

    # 2. Campus Exterior Summer
    p2 = REF_DIR / "ref_campus_exterior_radar_summer.jpg"
    if p2.is_file():
        img2 = Image.open(p2)
        samples.append((
            "campus_exterior_full",
            img2.resize((512, 288), Image.Resampling.LANCZOS),
            "<26063_pixel_style>, scientific meteorological station campus on rocky granite shore of Gulf of Finland, weather radar tower with white spherical radome, modern ochre facade building, large illuminated windows, stone staircase, lanterns, pine and birch forest, low horizon with St Petersburg silhouette, pixel art landscape"
        ))
        samples.append((
            "campus_radar_tower",
            img2.crop((550, 100, 1000, 550)).resize((512, 512), Image.Resampling.LANCZOS),
            "<26063_pixel_style>, weather radar tower building, spherical white radome dome, meteorological anemometer mast, ochre and concrete building, large warm glowing windows, granite stone stairs and retaining wall, birch trees, clear northern summer day, detailed pixel art building"
        ))
        samples.append((
            "campus_coastal_path",
            img2.crop((0, 150, 550, 576)).resize((512, 512), Image.Resampling.LANCZOS),
            "<26063_pixel_style>, granite rocky coastline, walking path on shore of Gulf of Finland, northern pine trees, boulders with green moss, warm lantern post, blue sea water, distant coastal buildings, sunny day, pixel art nature"
        ))

    # 3. Stormy Autumn Station
    p3 = REF_DIR / "ref_station_storm_autumn.jpg"
    if p3.is_file():
        img3 = Image.open(p3)
        samples.append((
            "station_storm_autumn_full",
            img3.resize((512, 288), Image.Resampling.LANCZOS),
            "<26063_pixel_style>, stormy autumn evening at coastal meteorological post, dark overcast cumulonimbus clouds, rain squall over rough sea waves, circular glass observation rotunda with warm light and working synopticians, cottage with red tile roof, orange autumn maple leaves on wet stone pavement, striped windsock, radar tower, lighthouse, moody atmospheric pixel art"
        ))
        samples.append((
            "station_observation_rotunda",
            img3.crop((40, 200, 420, 576)).resize((512, 512), Image.Resampling.LANCZOS),
            "<26063_pixel_style>, circular meteorological observation rotunda, panoramic glass bay window, warm interior light, observers analyzing synoptic charts, flat roof with weather radar dome and radio antennas, granite foundation, rough sea with foam in background, golden autumn foliage, pixel art"
        ))
        samples.append((
            "station_autumn_cottage_pavement",
            img3.crop((500, 150, 1024, 576)).resize((512, 512), Image.Resampling.LANCZOS),
            "<26063_pixel_style>, cozy weather station house, red clay tile roof, warm window with ginger cat looking outside, green entrance door with umbrella and rubber boots, fallen golden autumn leaves on wet reflective stone tiles, lantern light, stormy sky, detailed pixel art"
        ))
        samples.append((
            "station_sea_lighthouse_storm",
            img3.crop((350, 200, 800, 500)).resize((512, 512), Image.Resampling.LANCZOS),
            "<26063_pixel_style>, coastal lighthouse on rocky promontory in stormy sea, meteorological radar tower, windsock fluttering in high wind, breaking ocean waves with foam, dark storm clouds, autumn birch trees, pixel art scenery"
        ))

    # 4. Meteo Playground Summer
    p4 = REF_DIR / "ref_meteo_playground_summer.jpg"
    if p4.is_file():
        img4 = Image.open(p4)
        samples.append((
            "meteo_playground_full",
            img4.resize((512, 288), Image.Resampling.LANCZOS),
            "<26063_pixel_style>, standard Russian meteorological observation field, meteoploshchadka, white psychrometric louvered shelter будка Селянинова, Tretyakov precipitation gauge осадкомер Третьякова, Wild wind vane mast, white picket fence, synoptic campus main building with radar tower and aerological sounding balloon hangar in background, bench with synoptic map and bird, sunny summer, pixel art"
        ))
        samples.append((
            "meteo_instruments_field",
            img4.crop((560, 300, 960, 576)).resize((512, 512), Image.Resampling.LANCZOS),
            "<26063_pixel_style>, meteorological observation instruments on grassy field, white louvered thermometer shelter on high stilts, precipitation gauge cylinder, anemometer mast, white picket fence, small weather post cottage with red roof, birch trees, bright daylight, authentic Russian meteorological station, pixel art"
        ))
        samples.append((
            "meteo_radar_and_hangar",
            img4.crop((50, 50, 550, 550)).resize((512, 512), Image.Resampling.LANCZOS),
            "<26063_pixel_style>, tall red-brick radar tower with white spherical radome dome, aerological hangar with weather balloon sounding equipment, birch and pine trees, granite rocks, blue sky with puffy cumulus clouds, scientific meteorological campus, pixel art"
        ))
        samples.append((
            "meteo_bench_atlas",
            img4.crop((0, 350, 400, 576)).resize((512, 512), Image.Resampling.LANCZOS),
            "<26063_pixel_style>, wooden park bench with open synoptic weather map book, tit bird perched on backrest, granite boulders, wild rose bushes with red berries, birch tree trunk, sunny day, charming environmental pixel art"
        ))

    # 5. Characters from Image 5
    p5 = REF_DIR / "ref_station_storm_with_characters.jpg"
    if p5.is_file():
        img5 = Image.open(p5)
        # Senior female synoptician portrait
        p_woman = img5.crop((815, 0, 1024, 256))
        samples.append((
            "char_senior_female_synoptician",
            p_woman.resize((512, 512), Image.Resampling.NEAREST),
            "<26063_character>, <26063_pixel_style>, portrait of senior female chief synoptician, middle-aged woman professor with kind wise expression, round eyeglasses, gray hair tied in elegant bun, holding red and blue meteorological synoptic pencil, cozy burgundy knit cardigan, white collared blouse, green banker desk lamp in warm background, detailed character pixel art portrait"
        ))
        # Tight face crop female synoptician
        samples.append((
            "char_female_synoptician_face",
            p_woman.crop((20, 10, 190, 200)).resize((512, 512), Image.Resampling.NEAREST),
            "<26063_character>, <26063_pixel_style>, close-up face portrait of senior female meteorologist with round spectacles, intelligent kind eyes, smiling warmly, holding red blue synoptic pencil, elegant gray hair, pixel art avatar"
        ))

        # Senior male observer portrait
        p_man = img5.crop((815, 256, 1024, 512))
        samples.append((
            "char_senior_male_observer",
            p_man.resize((512, 512), Image.Resampling.NEAREST),
            "<26063_character>, <26063_pixel_style>, portrait of senior male meteorological station technician and marine observer, weathered friendly face, thick graying mustache and stubble, dark blue knit watch cap beanie, heavy warm olive waterproof parka coat with pockets holding pens and instruments, dark gray wool turtleneck sweater, detailed character pixel art portrait"
        ))
        # Tight face crop male observer
        samples.append((
            "char_male_observer_face",
            p_man.crop((20, 10, 190, 200)).resize((512, 512), Image.Resampling.NEAREST),
            "<26063_character>, <26063_pixel_style>, close-up face portrait of experienced male weather observer and sea forecaster, mustache, blue beanie hat, warm smile, weather-beaten cheerful face, pixel art avatar"
        ))

    # Add horizontal flips for landscape / architectural samples to enrich diversity
    augmented: list[tuple[str, Image.Image, str]] = []
    for name, im, cap in samples:
        augmented.append((name, im, cap))
        if not name.startswith("char_"):
            flip_im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            augmented.append((f"{name}_flip", flip_im, cap))

    for name, im, cap in augmented:
        im_path = OUT_DIR / f"{name}.png"
        txt_path = OUT_DIR / f"{name}.txt"
        im.save(im_path)
        txt_path.write_text(cap, encoding="utf-8")

    return augmented


if __name__ == "__main__":
    items = build_dataset()
    print(f"✅ Successfully prepared {len(items)} master training samples in {OUT_DIR}")
