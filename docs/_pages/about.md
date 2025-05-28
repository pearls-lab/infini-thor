---
layout: default
title: About
permalink: /
subtitle: 

abstract: >
  abstract
intro: >
  intro
---

<div class="text-center my-5">

  <!-- Author Names (Linked) -->
  <div style="font-size: 1.3rem; margin-bottom: 0.5rem; font-weight: bold;">
    <a href="http://bosung.github.io/" target="_blank">Bosung Kim</a> and
    <a href="https://prithvirajva.com" target="_blank">Prithviraj Ammanabrolu</a>
  </div>

  <!-- Affiliation -->
  <div style="font-size: 1.3rem; color: #555; font-weight: bold;">
    UC San Diego
  </div>
</div>

<div style="text-align: center; margin-top: 2rem;">
  <a href="https://arxiv.org/pdf/2505.16928" target="_blank" style="text-decoration: none;">
    <button style="margin: 0.5rem; padding: 0.7rem 1.2rem; font-size: 1rem; border-radius: 12px; border: none; background-color: #b31b1b; color: white;">
      📄 Paper
    </button>
  </a>
  <a href="https://github.com/pearls-lab/infini-thor" target="_blank" style="text-decoration: none;">
    <button style="margin: 0.5rem; padding: 0.7rem 1.2rem; font-size: 1rem; border-radius: 12px; border: none; background-color: #24292e; color: white;">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" alt="GitHub" style="height: 1.2em; vertical-align: middle; margin-right: 0.5em;">
      Code
    </button>
  </a>
  <a href="https://huggingface.co/datasets/PEARLS-Lab/infini-thor" target="_blank" style="text-decoration: none;">
    <button style="margin: 0.5rem; padding: 0.7rem 1.2rem; font-size: 1rem; border-radius: 12px; border: none; background-color: #ffcc00; color: black;">
      <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HF" style="height: 1.2em; vertical-align: middle; margin-right: 0.5em;">
      Dataset
    </button>
  </a>
</div>

<div style="margin-bottom: 5rem;"></div>

<div class="text-center my-5" style="margin-bottom: 5rem;">
  <h3 style="font-size: 2.0rem; font-weight: bold;">
    Abstract
  </h3>
  <p class="abstract-text">
  We introduce <span>\(\infty\)</span>-THOR, a new framework for long-horizon embodied tasks that advances long-context understanding in embodied AI.
<span>\(\infty\)</span>-THOR provides:
</p>
<p class="abstract-text">
(1) a generation framework for synthesizing scalable, reproducible, and unlimited long-horizon trajectories;
(2) a novel embodied QA task, Needle(s) in the Embodied Haystack, where multiple scattered clues across extended trajectories test agents' long-context reasoning ability; and
(3) a long-horizon dataset and benchmark suite featuring complex tasks that span hundreds of environment steps, each paired with ground-truth action sequences.
To enable this capability, we explore architectural adaptations, including interleaved Goal-State-Action modeling, context extension techniques, and Context Parallelism, to equip LLM-based agents for extreme long-context reasoning and interaction.
Experimental results and analyses highlight the challenges posed by our benchmark and provide insights into training strategies and model behaviors under long-horizon conditions.
Our work provides a foundation for the next generation of embodied AI systems capable of robust, long-term reasoning and planning.
</p>
</div>

<div style="margin-bottom: 10rem;"></div>

<div class="text-center my-5" style="margin-bottom: 5rem;">
  <h3 style="font-size: 2.0rem; font-weight: bold;">Demo Video</h3>

  <video controls autoplay style="max-width: 800px; width: 100%; margin: 0 auto; display: block;">
    <source src="assets/video/intro_vid.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <p class="abstract-text" style="margin-top: 20px; margin-bottom: 20px;">
  Our generation framework can generate unlimited tasks, the trajectories can be exceptionally long, exceeding 1M context tokens or beyond.
  </p>

  <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
    <video controls autoplay muted loop style="max-width: 240px; width: 100%;">
      <source src="assets/video/floorplan323_19_932.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <video controls autoplay muted loop style="max-width: 240px; width: 100%;">
      <source src="assets/video/floorplan218_17_889.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <video controls autoplay muted loop style="max-width: 240px; width: 100%;">
      <source src="assets/video/floorplan210_26_870.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>
</div>

<div style="margin-bottom: 10rem;"></div>

<div class="text-center my-5" style="margin-bottom: 5rem;">
  <h3 style="font-size: 2.0rem; font-weight: bold;">Needle(s) in the Emboded Haystack</h3>
  <!-- <h3>Needle(s) in the Emboded Haystack</h3> -->

  <!-- <h3 style="
      background-image: linear-gradient(rgba(255, 255, 255, 0.6), rgba(255, 255, 255, 0.6)), url('{{ site.baseurl }}/assets/img/background-t7.png');
      background-size: cover;
      background-position: center;
      padding: 1.5rem;
      border-radius: 8px;
      color: black;
      font-size: 2.0rem;
      font-weight: bold;
      text-align: center;
    ">Needle(s) in the Emboded Haystack</h3> -->
  <div class="text-center my-5">
    <p class="main-text">
    <span>\(\infty\)</span>-THOR introduces a new challenging task, Needle(s) in the Embodied Haystack (NiEH).
    Unlike the standard Needle in a Haystack task, which focuses on recalling a single clue in text, NiEH poses two main challenges:
    <span style="font-weight: bold;">(1) multiple scattered clues (Needles)</span> and <span style="font-weight: bold;">(2) multi-modal inputs that combine visual and linguistic observations from the environment (Embodiment)</span>.
    This task is designed to evaluate the agent's ability to recall and reason about previously encountered environmental details, such as identifying objects and recalling performed actions.

    Figure 3 and 4 present examples of the two NiEH task types.
    In the single-evidence setting, a question is answerable based on a single observation step; in the multi-evidence setting, multiple temporally distant steps must be combined to answer the question.
  </p>
  </div>

  <figure style="max-width: 1000px; margin: 0 auto 40px; text-align: center;">
    <img src="assets/img/example_NiEH.png" alt="First Image" style="width: 100%; height: auto; display: block; margin: 0 auto;">
    <figcaption style="margin-top: 10px; font-size: 1.1rem; color: #555;">
       <span style="font-weight: bold;">Figure 1.</span> Example of Needle in the Embodied Haystack: Single-evidence question types.
    </figcaption>
  </figure>

  <figure style="max-width: 1000px; margin: 0 auto; text-align: center;">
    <img src="assets/img/example_NiSSSEH.png" alt="Second Image" style="width: 100%; height: auto; display: block; margin: 0 auto;">
    <figcaption style="margin-top: 10px; font-size: 1.1rem; color: #555;">
       <span style="font-weight: bold;">Figure 2.</span> Example of Needles in the Embodied Haystack: Multi-evidence question types.
    </figcaption>
  </figure>
</div>

<div style="margin-bottom: 10rem;"></div>

<div class="text-center my-5" style="margin-bottom: 5rem;">
  <h3 style="font-size: 2.0rem; font-weight: bold;">Long-horizon Trajectories for Interactive Evaluations</h3>
  <!-- <h3>Long-horizon Trajectories for Interactive Evaluations</h3> -->

  <div class="text-center my-5">
    <p class="main-text">
    Our benchmark uniquely features tasks with a synthetic final goal, which involves multiple objects that appear at distant time steps,
    requiring multi-step reasoning across over hundreds of steps.
    Figure 3 illustrates an example: the agent observes the tomato at an early step (t=17) and the counter top much later (t=560). Then, the final task is given at t=670, which requires the agent to place the tomato on the counter top.
This setup highlights the challenge of long-horizon dependency, where key objects and locations must be remembered and acted upon after hundreds of steps.
  </p>
  </div>

  <figure style="max-width: 1000px; margin: 0 auto 40px; text-align: center;">
    <img src="assets/img/long-horizon.png" alt="First Image" style="width: 100%; height: auto; display: block; margin: 0 auto;">
    <figcaption style="margin-top: 10px; font-size: 1.1rem; color: #555;">
     <span style="font-weight: bold;">Figure 3.</span> Example of the trajectory and a long-horizon embodied task generated from <span>\(\infty\)</span>-THOR. The
final goal (“Put the tomato on the counter top” at t=670) requires recalling both the tomato (seen at
t=17) and the counter (seen at t=560) to solved the long-horizon task. Context size refers to the input
token length when converting the trajectory into the LLM input space.
    </figcaption>
  </figure>
</div>