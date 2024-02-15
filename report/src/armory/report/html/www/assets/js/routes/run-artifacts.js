import { storeToRefs } from 'pinia';
import { computed } from 'vue';
import { useArtifactSettings } from '../stores/artifact-settings.js';

export default {
    props: {
        run: Object,
    },
    setup(props) {
        const keys = computed(() => {
            const chains = new Set();
            const batches = new Set();
            const samples = new Set();
            for (const [chain, chain_artifacts] of Object.entries(props.run.artifacts)) {
                chains.add(chain);
                for (const [batch, batch_artifacts] of Object.entries(chain_artifacts)) {
                    batches.add(batch);
                    for (const sample of Object.keys(batch_artifacts)) {
                        samples.add(sample);
                    }
                }
            }
            return {
                chains: [...chains],
                batches: [...batches],
                samples: [...samples],
            }
        });

        const { batch, lhsChain, rhsChain, sample } = storeToRefs(useArtifactSettings());

        const lhsImage = computed(() => {
            const file = props.run.artifacts[lhsChain.value]?.[batch.value]?.[sample.value];
            if (file) {
                return `./assets/img/${props.run.info.run_id}/${file}`;
            }
            return "";
        });

        const rhsImage = computed(() => {
            const file = props.run.artifacts[rhsChain.value]?.[batch.value]?.[sample.value];
            if (file) {
                return `./assets/img/${props.run.info.run_id}/${file}`;
            }
            return "";
        });

        return {
            batch,
            keys,
            lhsChain,
            lhsImage,
            rhsChain,
            rhsImage,
            sample,
        };
    },
    template: `
        <div class="flex gap-2 items-center my-2">
            <span>Chains</span>
            <select v-model="lhsChain" class="select select-bordered select-sm w-60">
                <option disabled value="">First chain?</option>
                <option v-for="chain in keys.chains">
                    {{ chain }}
                </option>
            </select>
            <span>vs</span>
            <select v-model="rhsChain" class="select select-bordered select-sm w-60">
                <option disabled value="">Second chain?</option>
                <option v-for="chain in keys.chains">
                    {{ chain }}
                </option>
            </select>
            <span class="border-l-2 pl-2">
                Batch
            </span>
            <select v-model="batch" class="select select-bordered select-sm w-40">
                <option disabled value="">Which batch?</option>
                <option v-for="batch in keys.batches">
                    {{ batch }}
                </option>
            </select>
            <span class="border-l-2 pl-2">
                Sample
            </span>
            <select v-model="sample" class="select select-bordered select-sm w-40">
                <option disabled value="">Which sample?</option>
                <option v-for="sample in keys.samples">
                    {{ sample }}
                </option>
            </select>
        </div>
        <div
            v-if="lhsImage && rhsImage"
            class="diff aspect-square mx-auto max-w-xl"
        >
            <div class="diff-item-1">
                <img :src="rhsImage" />
            </div>
            <div class="diff-item-2">
                <img :src="lhsImage" />
            </div>
            <div class="diff-resizer"></div>
        </div>
    `,
};
