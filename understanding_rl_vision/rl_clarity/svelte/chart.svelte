<script>
  export let values = null;
  export let colors = null;
  export let extra_values = null;
  export let disable_extra = false;
  export let state;
  export let height;
  export let width;
  export let bar_width_frac = 1;
  export let quantile_to_overflow = 0.9;
  export let max_overflow = 5;
  export let ticks_width_frac = 0.05;
  export let ticks_count = 5;
  export let ticks_overlapped = false;

  const get_reduced_values_asc = function(values, extra_values, reduce) {
    if (values === null) {
      return null;
    }
    else {
      let reduced_values = values.map(arr => reduce.apply(null, arr));
      if (extra_values !== null) {
        reduced_values = reduced_values.map((value, i) =>
          reduce(value, extra_values[i]));
      }
      reduced_values.sort((a, b) => a - b);
      return reduced_values;
    }
  };
  $: max_values_asc = get_reduced_values_asc(values, extra_values, Math.max);
  $: min_values_asc = get_reduced_values_asc(values, extra_values, Math.min);
  const quantile = function(arr_asc, q) {
    let pos = (arr_asc.length - 1) * q;
    let base = Math.floor(pos);
    let rest = pos - base;
    if (typeof(arr_asc[base + 1]) === "undefined") {
      return arr_asc[base];
    }
    else {
      return arr_asc[base] + rest * (arr_asc[base + 1] - arr_asc[base]);
    }
  };
  $: upper_value = (function() {
    if (values === null) {
      return null;
    }
    else {
      return Math.max(
        0, quantile(max_values_asc, quantile_to_overflow), Math.max.apply(null, max_values_asc) / max_overflow);
    }
  })();
  $: lower_value = (function () {
    if (values === null) {
      return null;
    }
    else {
      return Math.min(
        0, quantile(min_values_asc, 1 - quantile_to_overflow), Math.min.apply(null, min_values_asc) / max_overflow);
    }
  })();
    
  $: [values_to_display, colors_to_display, outlines_to_display] = (function() {
    if (values === null || upper_value === lower_value) {
      return [null, null, null];
    }
    else {
      let values_to_display = values[state.position].slice();
      let colors_to_display = colors.slice();
      let outlines_to_display = colors.map(() => "none");
      if (extra_values !== null) {
        values_to_display.push(extra_values[state.position]);
        colors_to_display.push("white");
        outlines_to_display.push("1px " + (disable_extra ? "dashed" : "solid") + " gray");
      }
      return [values_to_display, colors_to_display, outlines_to_display];
    }
  })();
</script>

{#if values_to_display !== null}
  <div style="position: relative; height: {height}; width: {width}; padding: 0px 1px;">
    <div style="position: absolute;
                top: 0%;
                left: 0%;
                height: 100%;
                width: {ticks_width_frac * 100}%;"
    >
      <div class="center-text"
           style="position: absolute;
                  z-index: 2;
                  top: {((upper_value / (upper_value - lower_value)) - 0.5) * 100}%;
                  left: 0%;
                  height: 100%;
                  width: 50%;"
      >0</div>
      {#each Array.from(Array(ticks_count).keys()) as tick_number}
        <div style="position: absolute;
                    z-index: 3;
                    top: {(((upper_value / (upper_value - lower_value)) + (tick_number / ticks_count)) % 1) * 100}%;
                    left: 75%;
                    height: 1px;
                    width: 25%;
                    background-color: gray;"
        ></div>
      {/each}
      <div style="position: absolute;
                  z-index: 3;
                  top: 0%;
                  left: 100%;
                  height: 100%;
                  width: 1px;
                  background-color: gray;"
      ></div>
    </div>
    <div style="position: absolute;
                top: 0%;
                  left: {(ticks_overlapped ? 0 : ticks_width_frac) * 100}%;
                  height: 100%;
                  width: {(ticks_overlapped ? 1 : (1 - ticks_width_frac)) * 100}%;"
    >
      {#each values_to_display as value, index}
        <div style="position: absolute;
                    z-index: 2;
                    top: {((upper_value - Math.max(value, 0)) / (upper_value - lower_value)) * 100}%;
                    left: {((index + (1 - bar_width_frac) / 2) / values_to_display.length) * 100}%;
                    height: {(Math.abs(value) / (upper_value - lower_value)) * 100}%;
                    width: {(bar_width_frac / values_to_display.length) * 100}%;
                    background-color: {colors_to_display[index]};
                    outline: {outlines_to_display[index]};
                    outline-offset: -1px;"
        ></div>
      {/each}
      <div style="position: absolute;
                  z-index: 3;
                  top: {(upper_value / (upper_value - lower_value)) * 100}%;
                  left: {(ticks_overlapped ? ticks_width_frac : 0) * 100}%;
                  height: 1px;
                  width: {(ticks_overlapped ? (1 - ticks_width_frac) : 1) * 100}%;
                  background-color: gray;"
      ></div>
      {#each Array.from(Array(ticks_count).keys()) as tick_number}
        {#if tick_number !== 0}
          <div style="position: absolute;
                      z-index: 1;
                      top: {(((upper_value / (upper_value - lower_value)) + (tick_number / ticks_count)) % 1) * 100}%;
                      left: {(ticks_overlapped ? ticks_width_frac : 0) * 100}%;
                      height: 1px;
                      width: {(ticks_overlapped ? (1 - ticks_width_frac) : 1) * 100}%;
                      background-color: lightgray;"
          ></div>
        {/if}
      {/each}
    </div>
  </div>
{/if}
