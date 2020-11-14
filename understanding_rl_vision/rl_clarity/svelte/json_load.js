// reduced version of lucid/scratch/js/src/load.js

const active_requests = new Map();
const cache = new Map();

const handle_errors = function(response) {
  if (response.ok) {
    return response;
  } else {
    throw new Error(response.status + ':' + response.statusText);
  }
};

const json_loader = function(url, json_preloaded) {
  if (typeof(json_preloaded) !== "undefined" && typeof(json_preloaded[url]) !== "undefined") {
    return new Promise((resolve) => {
      resolve(json_preloaded[url]);
    });
  }
  else if (cache.has(url)) {
    return cache.get(url);
  }
  else {
    let promise = fetch(url).then(handle_errors).then(response => response.json());
    cache.set(url, promise);
    return promise;
  }
};


export const json_load = function(url, namespace, json_preloaded) {
  let request_id = 0;
  if (typeof(namespace) !== "undefined") {
    if (active_requests.has(namespace)){
      request_id = active_requests.get(namespace) + 1;
    }
    active_requests.set(namespace, request_id);
  }
  return new Promise((resolve, reject) => {
    let promise;
    if (Array.isArray(url)) {
      promise = Promise.all(url.map((u) => json_loader(u, json_preloaded)));
    }
    else {
      promise = json_loader(url, json_preloaded);
    }
    promise.then((response) => {
      if (typeof(namespace) === "undefined" || active_requests.get(namespace) === request_id) {
        resolve(response);
      }
    }).catch((error) => {
      if (typeof(namespace) === "undefined" || active_requests.get(namespace) === request_id) {
        reject(error);
      }
    });
  });
};
