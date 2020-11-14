<script>
  import Scrubber from './scrubber.svelte';

  export let titles = [];
  export let series = [];
  export let dones = [];
  export let colors = [];
  export let state;
  export let height;
  export let width;
  export let visible_duration = 1;
  export let scrubber = visible_duration > 1;
  export let label_precision = 3;

  let graph_div;
  $: vertical_scale = (function () {
    let length_ratio = parseFloat(height) / parseFloat(width);
    if (typeof(graph_div) !== "undefined") {
      let suffix = (s => s.match(/[^0-9]*$/)[0].trim());
      if (suffix(height) !== suffix(width)) {
        length_ratio = graph_div.offsetHeight / graph_div.offsetWidth;
      }
    }
    return 0.98 * visible_duration * length_ratio;
  })();
  $: series_rescaled = series.map(function(l) {
    let min = Math.min.apply(null, l);
    let max = Math.max.apply(null, l);
    return l.map(x => min == max ? 0 : vertical_scale * (x - min) / (max - min));
  });
  $: max_duration = Math.max.apply(null, series.map(arr => arr.length));
  $: svg_display_for_refresh = (function() {
    if (state.position % 2 == 0) {
      return "block";
    }
    else {
      return "initial";
    }
  })();

  const format_number = function(n, precision = label_precision) {
    let base = n.toPrecision(precision).replace("-", "&minus;");
    let exponent = "";
    if (base.includes("e")) {
      [base, exponent] = base.split("e");
    }
    base = base.replace(/\.0+$/, "");
    if (exponent !== "") {
      return base + "&times;10" + "<sup>" + exponent + "</sup>";
    }
    else {
      return base;
    }
  };
</script>

<style>
 .container {
   position: absolute;
   overflow: hidden;
   height: 100%;
   width: 100%;
 }

 line {
   stroke-width: 0.5;
 }

 .label {
   background-color: rgba(255, 255, 255, 0.9);
   font-weight: bold;
 }
</style>

<div bind:this={graph_div} style="position: relative; height: {height}; width: {width};">
  <div class="container" style="z-index: 0;">
    <svg
      viewBox="0 -{vertical_scale} {max_duration} {vertical_scale}"
      style="display: {svg_display_for_refresh};
             position: absolute;
             height: 100%;
             width: {(max_duration / visible_duration) * 100}%;
             left: {(0.5 - (state.position + 0.5) / visible_duration) * 100}%;
             background: white;
             outline: 2px solid gray;
             outline-offset: -1px;"
    >
      {#each Array.from(series.keys()) as index}
        {#each Array.from(Array(series[index].length - 1).keys()) as position}
          {#if !dones[index][position]}
            <line
              x1={position + 0.5}
              y1={- series_rescaled[index][position]}
              x2={position + 1.5}
              y2={- series_rescaled[index][position + 1]}
              style="stroke: {colors[index]};"
            ></line>
          {:else}
            <line
              x1={position + 1}
              y1={0}
              x2={position + 1}
              y2={- vertical_scale}
              style="stroke: lightgray;"
            ></line>
          {/if}
        {/each}
      {/each}
    </svg>
  </div>
  <div class="container" style="z-index: 1;">
    {#each Array.from(series.keys()) as index}
      {#if typeof(series[index][state.position]) !== "undefined"}
        <div
          class="label"
          style="position: absolute;
                 bottom: {((series.length - index - 0.5) / series.length) * 100}%;
                 left: 3px;
                 margin-bottom: -0.75em;
                 padding: 0.25em;
                 color: {colors[index]};"
        >
          {@html titles[index]}
        </div>
        <div
          class="label"
          style="position: absolute;
                 bottom: {((series.length - index - 0.5) / series.length) * 100}%;
                 right: calc(50% + 8px);
                 margin-bottom: -0.75em;
                 padding: 0.25em;"
        >
          {@html format_number(series[index][state.position])}
        </div>
      {/if}
    {/each}
  </div>
  <div class="container" style="z-index: 2;">
    {#if scrubber}
      <Scrubber bind:state={state} visible_duration={visible_duration} max_duration={max_duration}/>
    {/if}
  </div>
</div>
