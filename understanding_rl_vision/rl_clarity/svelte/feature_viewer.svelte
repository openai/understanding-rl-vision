<script>
  import { json_load } from './json_load.js';
  import { createEventDispatcher } from 'svelte';

  export let layer = null;
  export let number = null;
  export let image_dir = "";
  export let images = null;
  export let overlay_image_dir = null;
  export let overlay_image_fn = (overlay_image => overlay_image.toString() + ".png");
  export let overlay_image_grids_dir = null;
  export let overlay_image_grids_jsons = null;
  export let metadata = null;
  export let metadata_initial_values = {};
  export let json_preloaded = {};
  export let height;
  export let width;

  const dispatch = createEventDispatcher();

  const compareFunction = function(a, b) {
    let a_type = typeof(a);
    let b_type = typeof(b);
    if (a_type != b_type) {
      return a_type.localeCompare(b_type);
    }
    if (typeof(a) === "number") {
      return a - b;
    }
    return (a + "").localeCompare((b + "").toString());
  };

  $: metadata_configs = (function() {
    let configs = [{
      "key": "expand_mult",
      "text": ["zoom in", "zoom out"],
      "initial_value": 4,
      "scrollable": true,
    }, {
      "key": "subdiv_mult",
      "text": ["fewer patches", "more patches"],
      "initial_value": 0.5,
      "scrollable": false,
    }];
    for (let config_index = 0; config_index < configs.length; config_index++) {
      let config = configs[config_index];
      if (Object.prototype.hasOwnProperty.call(metadata_initial_values, config.key)) {
        config.initial_value = metadata_initial_values[config.key];
      }
      if (metadata === null) {
        config.values = []
      }
      else {
        config.values = Array.from(new Set(metadata[config.key])).sort(compareFunction);
      }
      let current_value;
      if (typeof(metadata_configs) !== "undefined") {
        let current_index = parseInt(metadata_configs[config_index].current_index);
        current_value = metadata_configs[config_index].values[current_index];
      }
      if (typeof(current_value) === "undefined") {
        current_value = config.initial_value;
      }
      config.current_index = config.values.indexOf(current_value);
      config.current_index = Math.max(0, config.current_index).toString();
    }
    return configs;
  })();
  $: selected_index = (function() {
    if (images !== null && metadata !== null) {
      for (let index = 0; index < images.length; index++) {
        let found = true;
        for (let config of metadata_configs) {
          if (metadata[config.key][index] !== config.values[parseInt(config.current_index)]) {
            found = false;
          }
        }
        if (found) {
          return index;
        }
      }
    }
    return null;
  })();

  const scroll = (function() {
    const lockout = 100;
    let prev_update_time = performance.now();
    return function(event) {
      let curr_time = performance.now();
      if (curr_time - prev_update_time > lockout) {
        for (let config of metadata_configs) {
          if (config.scrollable) {
            let old_value = parseInt(config.current_index);
            let new_value = old_value;
            new_value = old_value + Math.sign(event.deltaY);
            new_value = Math.max(0, Math.min(config.values.length - 1, new_value));
            new_value = new_value.toString();
            if (new_value !== old_value) {
              config.current_index = new_value;
              metadata_configs = (x => x)(metadata_configs);
              prev_update_time = curr_time;
            }
          }
        }
      }
    };
  })();

  let overlay_position = null;
  $: overlay_image_grids = (function() {
    if (overlay_image_grids_jsons !== null) {
      let urls = overlay_image_grids_jsons.map(json => overlay_image_grids_dir + json);
      json_load(urls, "features_grids", json_preloaded).then((grids) => {
        overlay_image_grids = grids;
        overlay_preload_secondary();
      });
    }
    return null;
  })();
  $: overlay_image = (function() {
    if (overlay_image_grids === null || selected_index === null || overlay_position === null) {
      return null;
    }
    else {
      let image_grid = overlay_image_grids[selected_index];
      let y_index = Math.floor(overlay_position.y * image_grid.length);
      y_index = Math.min(image_grid.length - 1, y_index);
      let image_array = image_grid[y_index];
      let x_index = Math.floor(overlay_position.x * image_array.length);
      x_index = Math.min(image_array.length - 1, x_index);
      return image_array[x_index];
    }
  })();
  const overlay_preload = function() {
    if (overlay_image_grids !== null && selected_index !== null) {
      let image_grid = overlay_image_grids[selected_index];
      for (let image_array of image_grid) {
        for (let image of image_array) {
          (new Image()).src = overlay_image_dir + overlay_image_fn(image);
        }
      }
    }
  };
  const overlay_preload_secondary = function() {
    if (overlay_position !== null) {
      overlay_preload();
    }
  };
  const overlay_update = function(event) {
    if (event.buttons === undefined ? event.which === 1 : event.buttons === 1) {
      if (overlay_position === null) {
        overlay_preload();
      }
      overlay_position = {
        "x": Math.max(0, Math.min(1, event.offsetX / event.target.scrollWidth)),
        "y": Math.max(0, Math.min(1, event.offsetY / event.target.scrollHeight))
      }
    }
    else {
      overlay_position = null;
    }
  };
</script>

<style>
 .image {
   position: absolute;
   background-size: 100% 100%;
   image-rendering: pixelated;
   height: 100%;
   width: 100%;
 }

 .label {
   white-space: nowrap;
 }

 td {
   vertical-align: top;
 }
</style>

<svelte:window on:mouseup={overlay_update}/>

<div
  class="striped"
  style="position: relative;
         overflow: hidden;
         height: {height};
         width: {width};
         border: 1px solid gray;"
  on:wheel|preventDefault={scroll}
  on:mouseover={overlay_update}
  on:mouseout={overlay_update}
  on:mousedown|preventDefault={overlay_update}
  on:mousemove={overlay_update}
>
  {#if images === null}  
    <div class="center-text" style="height: 100%; width: 100%;">Select a feature</div>
  {:else}
    {#each images as image, index}
      <div
        class="image opaque-hover"
        style="background-image: url('{image_dir + image}');
               visibility: {index === selected_index && overlay_image === null ? 'visible' : 'hidden'};
               cursor: pointer;"
      ></div>
    {/each}
    <div
      class="image"
      style="background-image: {overlay_image === null ? 'none' : 'url(\'' + overlay_image_dir + overlay_image_fn(overlay_image) + '\')'};
             visibility: {overlay_image === null ? 'hidden' : 'visible'};
             cursor: pointer;"
    ></div>
  {/if}
</div>

<h3 class="underrule">Feature visualization</h3>

<table style="width: 100%;">
  {#each metadata_configs as config, config_index}
    <tr>
      {#if config_index === 0 && layer !== null && number !== null}
        <td rowspan={metadata_configs.length + 1}>
          Layer {layer}, feature {number + 1}<br>
          Dataset examples by spatial position<br>
          Click to view example, scroll to zoom<br>
        </td>
      {/if}
      <td class="label" style="text-align: right;">{config.text[0]}</td>
      <td>
        <input
          type="range"
          min="0"
          max={Math.max(0, config.values.length - 1)}
          step="1"
          value={config.current_index}
          on:change={(event) => config.current_index = event.currentTarget.value}
          on:input={(event) => config.current_index = event.currentTarget.value}
        >
      </td>
      <td class="label">{config.text[1]}</td>
      <!-- <td>{#if config.scrollable}<small>[scrollable]</small>{/if}</td> -->
    </tr>
  {/each}
  <tr>
    <td colspan="3" style="text-align: right;">
      <button on:click={() => dispatch('close')}>close</button>
    </td>
  </tr>
</table>
