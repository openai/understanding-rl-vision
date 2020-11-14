<script>
  import Scrubber from './scrubber.svelte';

  export let image_dir = "";
  export let images;
  export let durations;
  export let weights = images.map(() => 1);
  export let grayscales = images.map(() => false);
  export let state;
  export let height;
  export let width;
  export let background_color = "black";
  export let visible_duration = 1;
  export let scrubber = visible_duration > 1;

  $: opacities = (function() {
    let opacities = [];
    let cum_weight = 0;
    for (let index = 0; index < weights.length; index++) {
      let weight = weights[index];
      if (durations !== null && state.position !== null) {
        if (state.position < 0 || state.position >= durations[index]) {
          weight = 0;
        }
      }
      cum_weight += weight;
      if (cum_weight === 0){
        opacities.push(0);
      }
      else {
        opacities.push(weight / cum_weight);
      }
    }
    return opacities;
  })();
  $: max_duration = Math.max.apply(null, durations);
</script>

<style>
 .container {
   position: absolute;
   overflow: hidden;
   height: 100%;
   width: 100%;
 }

 .image {
   position: relative;
   background-size: 100% 100%;
 }
</style>

<div style="position: relative; height: {height}; width: {width};">
  <div class="container" style="z-index: 0;">
    <div class="container" style="z-index: 0;">
      <div
        class="image pixelated"
        style="background-color: {background_color};
               height: 100%;
               width: {(durations[0] / visible_duration) * 100}%;
               left: {(0.5 - (state.position + 0.5) / visible_duration) * 100}%;
               outline: {scrubber ? '2px solid gray' : 'none'};
               outline-offset: -1px;"
      ></div>
    </div>
    {#each images as image, index}
      <div class="container" style="z-index: {(index + 1).toString()};">
        <div
          class={"image pixelated" + (grayscales[index] ? " grayscale": "")}
          style="opacity: {opacities[index].toString()};
                 background-image: url('{image_dir + image}');
                 height: 100%;
                 width: {(durations[index] / visible_duration) * 100}%;
                 left: {(0.5 - (state.position + 0.5) / visible_duration) * 100}%;"
        ></div>
      </div>
    {/each}
  </div>
  {#if scrubber}
    <div class="container" style="z-index: 1;">
      <Scrubber bind:state={state} visible_duration={visible_duration} max_duration={max_duration}/>
    </div>
  {/if}
</div>
