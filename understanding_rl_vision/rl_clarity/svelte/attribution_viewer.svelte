<script>
  import AttributionSelector from './attribution_selector.svelte';
  import Screen from './screen.svelte';
  import Chart from './chart.svelte';
  import { css_multiply } from './css_manipulate.js';
  import { json_load } from './json_load.js';

  export let trajectory = null;
  export let subdirs = {
    trajectories: "trajectories/",
    attribution: "attribution/",
    attribution_totals: "attribution_totals/"
  };
  export let images = null;
  export let metadata = null;
  export let channel_totals_jsons = null;
  export let residual_totals_jsons = null;
  export let totals_metadata = null;
  export let json_preloaded = {};
  export let state;
  export let attribution_kinds = [];
  export let initial_attribution_kinds = [];
  export let action_htmls = null;
  export let action_groups = null;
  export let max_duration;
  export let video_height;
  export let video_width;
  export let attribution_weight;
  export let attribution_or_gradients = "attribution";
  export let attribution_residual;
  export let attribution_policy = null;
  export let attribution_single_channel = null;
  export let channel_colors;

  const shallow_copy_object = function(obj) {
    if (obj === null) {
      return null;
    }
    else {
      return Object.assign({}, obj);
    }
  };

  const shallow_copy_list_of_objects = function(arr){
    let result = [];
    for (let value of arr) {
      result.push(shallow_copy_object(value));
    }
    return result;
  };

  const append_last_non_null = function(arr, default_value) {
    let new_value = default_value;
    for (let index = arr.length - 1; index >= 0; index--) {
      if (arr[index] !== null) {
        new_value = arr[index];
        break;
      }
    }
    return arr.concat([new_value]);
  };

  attribution_kinds = shallow_copy_list_of_objects(initial_attribution_kinds);
  /* let attribution_options = (function() {
   *   let attribution_options = [];
   *   for (let kind_index = 0; kind_index < attribution_kinds.length; kind_index++) {
   *     attribution_options.push(shallow_copy_list_of_objects(init.attribution_options));
   *   }
   *   return attribution_options;
   * })();
   * let add_attribution_kind = function() {
   *   attribution_kinds = append_last_non_null(attribution_kinds, initial_attribution_kinds[0]);
   *   attribution_kinds[attribution_kinds.length - 1] = shallow_copy_object(
   *     attribution_kinds[attribution_kinds.length - 1]);
   *   attribution_options = append_last_non_null(attribution_options, initial_attribution_options);
   *   attribution_options[attribution_options.length - 1] = shallow_copy_list_of_objects(
   *     attribution_options[attribution_options.length - 1]);
   * };
   * let add_attribution_options = function(kind_index) {
   *   let options = attribution_options[kind_index];
   *   options = append_last_non_null(options, init.attribution_options[0]);
   *   options[options.length - 1] = shallow_copy_object(options[options.length - 1]);
   *   attribution_options[kind_index] = options;
   * }; */
  let attribution_show_trajectory = true;
  let attribution_abs = false;
  const new_attribution_options = function(attribution_show_trajectory, attribution_abs, attribution_residual, attribution_single_channel) {
    let channel = attribution_single_channel === null ? (attribution_residual ? "all" : "prin") : attribution_single_channel;
    return [{direction: "non", channel: channel, show_trajectory: true}].concat(
      attribution_abs ? [
        {direction: "abs", channel: channel, show_trajectory: attribution_show_trajectory}
      ] : [
        {direction: "pos", channel: channel, show_trajectory: attribution_show_trajectory},
        {direction: "neg", channel: channel, show_trajectory: attribution_show_trajectory}
      ]
    );
  };
  let add_attribution_kind = function() {
    attribution_kinds = append_last_non_null(attribution_kinds, initial_attribution_kinds[0]);
    attribution_kinds[attribution_kinds.length - 1] = shallow_copy_object(
      attribution_kinds[attribution_kinds.length - 1]);
    attribution_options = attribution_options.concat([
      new_attribution_options(attribution_show_trajectory, attribution_abs, attribution_residual, attribution_single_channel)
    ]);
  };
  $: attribution_options = (function() {
    let attribution_options = [];
    for (let kind_index = 0; kind_index < attribution_kinds.length; kind_index++) {
      attribution_options.push(
        new_attribution_options(attribution_show_trajectory, attribution_abs, attribution_residual, attribution_single_channel)
      );
    }
    return attribution_options;
  })();
  $: attribution_images = (function() {
    let attribution_images = [];
    for (let kind_index = 0; kind_index < attribution_kinds.length; kind_index++) {
      let attribution_kind = attribution_kinds[kind_index];
      if (attribution_kind === null) {
        attribution_images.push(null);
      }
      else {
        let kind_images = [];
        for (let options of attribution_options[kind_index]) {
          if (options === null) {
            kind_images.push(null);
          }
          else {
            let selected_metadata = Object.assign(Object.assign({}, attribution_kind), options);
            kind_images.push((function() {
              for (let image_index = 0; image_index < images.length; image_index++) {
                let found = true;
                for (let key in metadata) {
                  if (Object.prototype.hasOwnProperty.call(metadata, key)) {
                    if (metadata[key][image_index] !== selected_metadata[key]) {
                      found = false;
                    }
                  }
                }
                if (found) {
                  return images[image_index];
                }
              }
              return null;
            })());
          }
        }
        attribution_images.push(kind_images);
      }
    }
    return attribution_images;
  })();

  const load_attribution_totals = function(totals_jsons, callback, namespace) {
    if (totals_jsons !== null) {
      let urls = totals_jsons.map(json => subdirs.attribution_totals + json);
      json_load(urls, namespace, json_preloaded).then(callback);
    }
    return null;
  };
  const get_attribution_totals = function(totals, totals_metadata, attribution_kinds) {
    let attribution_totals = [];
    for (let kind_index = 0; kind_index < attribution_kinds.length; kind_index++) {
      let attribution_kind = attribution_kinds[kind_index];
      if (attribution_kind === null || totals == null) {
        attribution_totals.push(null);
      }
      else {
        let selected_metadata = Object.assign({}, attribution_kind);
        attribution_totals.push((function() {
          for (let totals_index = 0; totals_index < totals.length; totals_index++) {
            let found = true;
            for (let key in totals_metadata) {
              if (Object.prototype.hasOwnProperty.call(totals_metadata, key)) {
                if (totals_metadata[key][totals_index] !== selected_metadata[key]) {
                  found = false;
                }
              }
            }
            if (found) {
              return totals[totals_index];
            }
          }
          return null;
        })());
      }
    }
    return attribution_totals;
  };
  $: channel_totals = (function() {
    load_attribution_totals(channel_totals_jsons, function(totals) {
      channel_totals = totals;
    }, "attribution_channel_totals");
    return null;
  })();
  $: residual_totals = (function() {
    load_attribution_totals(residual_totals_jsons, function(totals) {
      residual_totals = totals;
    }, "attribution_residual_totals");
    return null;
  })();
  $: attribution_channel_totals = get_attribution_totals(channel_totals, totals_metadata, attribution_kinds);
  $: attribution_residual_totals = get_attribution_totals(residual_totals, totals_metadata, attribution_kinds);
</script>

<style>
 th, td {
   padding: 0.25em;
   border-bottom: 1px solid gray;
   text-align: left;
   vertical-align: top;
 }
</style>

<p>
  <label style="margin-right: 1em;"><input type="checkbox" bind:checked={attribution_show_trajectory}> overlay on observations</label>
  <label style="margin-right: 1em;"><input type="checkbox" bind:checked={attribution_abs}> combine positive and negative {attribution_or_gradients}</label>
  <label><input type="checkbox" bind:checked={attribution_residual}> show residual feature</label>
</p>

<table style="border-collapse: collapse;">
  <tr>
    {#if attribution_policy}
      <th colspan="1"></th>
    {/if}
    <th>Observation</th>
    {#if attribution_abs}
      <th>Positive and negative attribution</th>
    {:else}
      <th>Positive attribution</th>
      <th>Negative attribution</th>
    {/if}
  </tr>
  {#each attribution_kinds as attribution_kind, kind_index}
    {#if attribution_kind !== null}
      <tr>
        {#if attribution_policy}
          <AttributionSelector
            bind:type={attribution_kind.type}
            bind:data={attribution_kind.data}
            action_htmls={action_htmls}
            action_groups={action_groups}
            max_width={video_width}
          />
        {/if}
        {#each attribution_options[kind_index] as options, options_index}
          {#if options !== null}
            <td rowspan="{attribution_policy ? 2 : 1}">
              <div style="display: inline-block; border: 1px solid gray;">
                <Screen
                  image_dir={""}
                  images={[subdirs.trajectories + trajectory + ".png"].concat(options.direction !== "non" ? [subdirs.attribution + attribution_images[kind_index][options_index]] : [])}
                  durations={[max_duration, max_duration]}
                  weights={[options.show_trajectory ? 1 - attribution_weight : 0].concat(options.direction !== "non" ? [attribution_weight] : [])}
                  grayscales={options.direction !== "non" ? [true, false] : [false]}
                  bind:state={state}
                  height={options_index === 0 ? css_multiply(video_height, 7/8) : video_height}
                  width={options_index === 0 ? css_multiply(video_width, 7/8) : video_width}
                />
              </div>
              <!-- <AttributionOptionsSelector
                   bind:show_trajectory={options.show_trajectory}
                   bind:direction={options.direction}
                   bind:channel={options.channel}
                   attribution_or_gradients={attribution_or_gradients}
                   width={video_width}
                   /> -->
              {#if options_index === 0}
                <br>
                <div style="display: inline-block;">
                  <Chart
                    values={attribution_channel_totals[kind_index]}
                    colors={channel_colors}
                    extra_values={attribution_residual_totals[kind_index]}
                    disable_extra={!(attribution_residual || attribution_single_channel == "res")}
                    bind:state={state}
                    height={css_multiply(video_height, 1/4)}
                    width={css_multiply(video_width, 7/8)}
                  /> <!-- {attribution_residual ? attribution_residual_totals[kind_index] : null} -->
                </div>
              {/if}
            </td>
          {/if}
        {/each}
        <!-- <td>
             <button on:click={() => add_attribution_options(kind_index)}>add</button>
             </td> -->
      </tr>
      {#if attribution_policy}
        <tr>
          <td colspan="1" style="vertical-align: middle;">
            <button on:click={() => {attribution_kinds[kind_index] = null; attribution_options[kind_index] = null;}}>
              remove row
            </button>
          </td>
        </tr>
      {/if}
    {/if}
  {/each}
  {#if attribution_policy}
    <tr>
      <td colspan="1" style="vertical-align: middle; border-bottom: 0px;">
        <button on:click={add_attribution_kind}>add row</button>
      </td>
    </tr>
  {/if}
</table>
