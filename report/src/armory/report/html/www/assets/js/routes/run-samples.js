import { storeToRefs } from 'pinia';
import { computed } from 'vue';
import ImageClassificationSample from '../components/image-classification-sample.js';
import ObjectDetectionSample from '../components/object-detection-sample.js';
import { useArtifactSettings } from '../stores/artifact-settings.js';
import { useEvaluationData } from '../stores/evaluation-data.js';

export default {
    components: {
        ImageClassificationSample,
        ObjectDetectionSample,
    },
    props: {
        run: Object,
    },
    setup(props) {
        const evaluationData = useEvaluationData();
        const taskSpecificComponent = computed(() => ({
            'image-classification': ImageClassificationSample,
            'object-detection': ObjectDetectionSample,
        }[evaluationData.settings.task]));

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

        const settings = storeToRefs(useArtifactSettings());

        return {
            ...settings,
            keys,
            taskSpecificComponent,
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
            <input
                v-model="sideBySide"
                id="side-by-side"
                type="checkbox"
            />
            <label for="side-by-side" class="hover:cursor-pointer">
                Side-by-side
            </label>
        </div>
        <component
            :is="taskSpecificComponent"
            :batch="batch"
            :lhsChain="lhsChain"
            :rhsChain="rhsChain"
            :run="run"
            :sample="sample"
            :sideBySide="sideBySide"
        ></component>
    `,
};