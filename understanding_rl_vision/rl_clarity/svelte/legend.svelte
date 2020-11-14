<script>
  import { css_multiply } from './css_manipulate.js';
  import { createEventDispatcher } from 'svelte';

  export let image_dir = "";
  export let image = null;
  export let colors;
  export let labels = colors.map(() => null);
  export let item_height;
  export let item_width;
  export let show_residual;
  export let selected_channel = null;
  export let enable_hover;

  const dispatch = createEventDispatcher();
</script>

<style>
 .container {
   position: absolute;
   overflow: hidden;
   height: 100%;
   width: 58.82%;
   top: 0%;
   left: 41.18%;
   border: 1px solid gray;
 }

 .image {
   position: relative;
   background-size: 100% 100%;
 }

 .label {
   font-weight: bold;
   color: white;
 }
</style>

{#each colors as color, index}
  <div style="padding: 0.5em;
              background: {selected_channel === index ? 'whitesmoke' : 'white'};
              border: 1px solid {selected_channel === index ? 'gray' : 'white'}; 
              border-radius: 0.25em;
              cursor: pointer;
              overflow: hidden;"
       on:click={(event) => {dispatch('select', index); event.stopPropagation();}}
       on:mouseover={() => {if (enable_hover) {selected_channel = index;}}}
       on:mouseout={() => {if (enable_hover) {selected_channel = null;}}}
  >
    <div
      style="position: relative;
             height: {item_height};
             width: {css_multiply(item_width, 1.7)};
             white-space: nowrap;"
    >
      <div
        class="center-text"
        style="position: absolute;
               top: 25%;
               left: 2.94%;
               height: 50%;
               width: 29.41%;
               background-color: {color};
               border-radius: 50%;
               color: white;
               font-weight: bold;"
      >{index + 1}</div>
      {#if image !== null}
        <div class="container striped" style="z-index: 0;">
          <div
            class="image pixelated opaque-hover"
            style="background-image: url('{image_dir + image}');
                   height: 100%;
                   width: {colors.length * 100}%;
                   left: {- index * 100}%;"
          ></div>
        </div>
      {/if}
      {#if labels[index] !== null}
        <div class="container label center-text" style="z-index: 1;">{labels[index]}</div>
      {/if}
    </div>
  </div>
{/each}

<div style="padding: 0.5em;
            background: {selected_channel === 'res' ? 'whitesmoke' : 'white'};
            border: 1px solid {selected_channel === 'res' ? 'gray' : 'white'}; 
            border-radius: 0.25em;
            cursor: pointer;
            overflow: hidden;"
     on:mouseover={() => {if (enable_hover) {selected_channel = "res";}}}
     on:mouseout={() => {if (enable_hover) {selected_channel = null;}}}
>
  <div
    style="position: relative;
           height: {item_height};
           width: {css_multiply(item_width, 1.7)};
           white-space: nowrap;"
  >
    <div
      class="center-text"
      style="position: absolute;
             top: 25%;
             left: 2.94%;
             height: 50%;
             width: 29.41%;
             font-size: 0.9em;
             font-weight: bold;
             color: gray;
             background-color: white;
             border: 1px {(show_residual || selected_channel == 'res') ? 'solid': 'dashed'} gray;
             border-radius: 50%;"
    >
      {#if !(show_residual || selected_channel == 'res')}
        not<br>shown
      {/if}
    </div>
    <div class="container center-text" style="font-style: italic; z-index: 1; background: white;">
      residual<br>(everything<br>else)
    </div>
  </div>
</div>
